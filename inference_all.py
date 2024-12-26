
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import PeftModel
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image, ImageEnhance

with open('tag_prediction.json') as f:
    ram_pred = json.load(f)
# Constants
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ALL_SAME_MODEL_PATH = ""

GENERAL_PATH = "./model/llava-fintune-yolo5"
REGIONAL_PATH = "./model/llava-finetune"
SUGGESTION_PATH = "./model/llava-finetune"

MAX_LENGTH = 800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans = True):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans
        self.bdd_model = YOLO("./model/bdd_best.pt")
        self.cone_model = YOLO("./model/cone_best.pt")

    def process_image_with_yolo(self, image):
        # Convert PIL Image to numpy array for YOLO
        image_np = np.array(image)
        
        # Get predictions from both models
        bdd_results = self.bdd_model(image_np)[0]
        cone_results = self.cone_model(image_np)[0]
        
        # Draw predictions on the image
        annotated_image = image_np.copy()
        
        # Define colors for different models
        colors = {
            'bdd100k': (0, 255, 0),  # Green for BDD100K
            'traffic_cone': (0, 0, 255)  # Red for traffic cones
        }
        
        # Process BDD100K predictions
        for result in bdd_results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         colors['bdd100k'], 
                         2)
        
        # Process traffic cone predictions
        for result in cone_results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         colors['traffic_cone'], 
                         2)
        
        # Convert back to PIL Image
        return Image.fromarray(annotated_image)
    
    def process_image_resolution(self, image):

        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        contrast_enhanced = contrast_enhancer.enhance(1.2)  # factor=1.2
        
        contrast_img_cv2 = cv2.cvtColor(np.array(contrast_enhanced), cv2.COLOR_RGB2BGR)
        
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_img_cv2, -1, kernel)
        
        final_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        
        return final_image
    
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        _, task, _ = sample["id"].split("_")
        original_image = sample["image"]

        if task == "regional":
            processed_image = original_image
            # processed_image = self.process_image_resolution(original_image) 
        else:
            # processed_image = original_image
            processed_image = self.process_image_with_yolo(original_image)

        conversation = sample["conversations"]
        if self.exist_ans:
            human_msg = conversation[0]["value"]
            assistant_msg = conversation[1]["value"]
            return sample["id"], task, processed_image, human_msg, assistant_msg
        else:
            human_msg = conversation[0]["value"]
            return sample["id"], task, processed_image, human_msg


def get_second_turn_prompt(first_response):
    return f"""Based on the previous response: {first_response} Please remove any repetitive statements and ensure every statement is completed."""



# Load models
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
    device_map=DEVICE
)
model = PeftModel.from_pretrained(base_model, GENERAL_PATH)
model.eval()

# Dataset and dataloader setup remains same
# [Your existing CodaDataset class and dataloader code]

# Improved prompt templates
prompt_template = 'USER: {} ASSISTANT:'
hf_dataset = {
    "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
}


custom_dataset = {
    "test": CodaDataset(hf_dataset["test"], exist_ans = False)
}

def collate_fn(batch):
    return zip(*batch)

test_loader = DataLoader(custom_dataset["test"], collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=0)
current_task = "general"
current_model = None
predictions = {}
with torch.no_grad():
    for sample_ids, tasks, images, human_msgs in tqdm(test_loader):
        sample_id = sample_ids[0]
        task = tasks[0]
        image = images[0]
        human_msg = human_msgs[0]
        
        if sample_id != "Test_general_0" and sample_id != "Test_regional_0" and sample_id != "Test_suggestion_0":
            continue
        
        if task != current_task:
            if task in ["regional"]:
                current_model = PeftModel.from_pretrained(base_model, REGIONAL_PATH)
            else:
                current_model = PeftModel.from_pretrained(base_model, SUGGESTION_PATH)
            current_task = task
            print(f"Switched to model for task: {task}")

        if task == "general":
            human_msg = f"<image>\nThis is an image of traffic captured from the perspective of the ego car. Here are the key elements in the image: {ram_pred[sample_id]}. Focus on objects in the image that directly influence the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.),  particularly those highlighted in rectangles. Pay special attention to traffic cones, barriers, and other highlighted objects. You must not discuss any objects beyond the seven categories above. For each highlighted object, describe its color, position, status, implications for driving, recommended responses, and its influence on the ego car's behavior."
        elif task == "regional":
            human_msg = "A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. EXPERT:"
        elif task == "suggestion":
            human_msg = "A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."

        prompts = [prompt_template.format(human_msg + ' Your comprehensive response should clear and accurate.')]
        inputs = processor(images=image, text=prompts, padding=True, return_tensors='pt').to(DEVICE, torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=MAX_LENGTH, do_sample=False) 
        
        generated_answer = processor.decode(outputs[0], skip_special_tokens=True).split('ASSISTANT: ')[1]
        print(f"Task: {task}, Generated answer for {sample_id}\n")

        predictions[sample_id] = generated_answer
        print(generated_answer)

        with open("submission_push.json", "w") as f:
            json.dump(predictions, f, indent=4)


