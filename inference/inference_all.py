
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import PeftModel
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image, ImageEnhance
import numpy as np
from ultralytics import YOLO
import cv2
with open('tag_prediction.json') as f:
    ram_pred = json.load(f)
# Constants
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER_PATH = "model_all/all-epoch-0"
MAX_LENGTH = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
# with open("submission_all.json", "r") as f:
#     predictions = json.load(f)

class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans = True, types_to_process = None):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans
        self.bdd_model = YOLO("bdd_best.pt")
        self.cone_model = YOLO("cone_best.pt")
        # if types_to_process is not None:
        #     self.hf_dataset = hf_dataset.filter(
        #         lambda x: x["id"].split("_")[1] in types_to_process
        # )
        #     # Print dataset size after filtering
        #     print(f"Filtered dataset size: {len(self.hf_dataset)}")
        # else:
        #     self.hf_dataset = hf_dataset
        
        # Print first few sample IDs
        print("First few sample IDs:")
        for i in range(min(5, len(self.hf_dataset))):
            print(self.hf_dataset[i]["id"])


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
    def brighten_image(self, image):

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
        
        # sample = self.hf_dataset[idx]
        sample = self.hf_dataset[idx]
        _, task, _ = sample["id"].split("_")
        original_image = sample["image"]

        if task in ["general", "suggestion"]:
            processed_image = self.process_image_with_yolo(original_image)
        else:
            processed_image = self.brighten_image(original_image)
        
        conversation = sample["conversations"]
        if self.exist_ans:
            human_msg = conversation[0]["value"]
            assistant_msg = conversation[1]["value"]
            return sample["id"], task, processed_image, human_msg, assistant_msg
        else:
            human_msg = conversation[0]["value"]
            return sample["id"], task, processed_image, human_msg

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
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Dataset and dataloader setup remains same
# [Your existing CodaDataset class and dataloader code]

# Improved prompt templates
prompt_template = 'USER: {} ASSISTANT:'
hf_dataset = {
    # "train": load_dataset("ntudlcv/dlcv_2024_final1", split="train"),
    # "val": load_dataset("ntudlcv/dlcv_2024_final1", split="val"),
    "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
}


custom_dataset = {
    # "train": CodaDataset(hf_dataset["train"], exist_ans = True),
    # "val": CodaDataset(hf_dataset["val"], exist_ans = True),
    "test": CodaDataset(hf_dataset["test"], exist_ans = False)
}
from torch.utils.data import Subset
total_samples = len(custom_dataset["test"])
subset_indices = {
    "test": range(total_samples - 5, total_samples)
}
subset_dataset = {
    "test": Subset(custom_dataset["test"], subset_indices["test"])
}
# subset_dataset = {
#     "test": Subset(custom_dataset["test"], subset_indices["test"])
# }
def collate_fn(batch):
    return zip(*batch)

test_loader = DataLoader(custom_dataset["test"], collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=0)

predictions = {}
with torch.no_grad():
    for sample_ids, tasks, images, human_msgs in tqdm(test_loader):
        sample_id = sample_ids[0]
        task = tasks[0]
        image = images[0]
        human_msg = human_msgs[0]
        if tasks[0] == "general":
            # observations = f"The scene contains the following elementss that may affect the ego car's behavior: {ram_pred[sample_id]}. Please describe the appearance, position, direction, and explain why it affects the ego car's behavior."
            # observations = f"This image contains the following elements that may affect the ego car's behavior: {ram_pred[sample_id]}. Please describe the appearance, position, direction, and explain why it affects the ego car's behavior."
            # observations = f"Here are some additional observations that we can see in the image: {ram_pred[sample_id]}. Please be aware of these objects and describe each object's appearance, position, direction, and explain why it affects the ego car's behavior"
           
            # human_msg = f"<image>\nThis is an image of traffic captured from the perspective of the ego car. Here are the key elements in the image: {ram_pred[sample_id]}. **Ensure that all these elements are included in your description.** For each element, provide details about its appearance, position, and direction. Explain how each element influences the ego car's driving behavior. If any element seems less important, still mention it and explain why it may or may not affect driving decisions."
            human_msg = (
                f"<image>\nThis is an image of traffic captured from the perspective of the ego car. "
                f"Here are the key elements in the image: {ram_pred[sample_id]}. Focus on objects in the "
                "image that directly influence the ego car's driving behavior: vehicles (cars, trucks, buses, "
                "braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic "
                "signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic "
                "cones, barriers, miscellaneous (debris, dustbin, animals, etc.), particularly those highlighted "
                "in rectangles. Pay special attention to traffic cones, barriers, and other highlighted objects. "
                "You must not discuss any objects beyond the seven categories above. For each highlighted object, "
                "describe its color, position, status, implications for driving, recommended responses, and its "
                "influence on the ego car's behavior."
            )
        elif tasks[0] == "regional":
            human_msg = (
            "<image>\nPlease describe the object inside the red "
            "rectangle in the image. Describe its color, position, status, implication, response, and explain why "
            "it affect ego car driving."
            )
        # human_msg = human_msg + "Please read the following message again: "+ human_msg
        elif tasks[0] == "suggestion":
            human_msg = (
                f"<image>\nThis is an image of traffic captured from the perspective of the ego car. "
                f"Here are the key elements in the image: {ram_pred[sample_id]}. Focus on objects in the "
                "image that directly influence the ego car's driving behavior: vehicles (cars, trucks, buses, "
                "braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic "
                "signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic "
                "cones, barriers, miscellaneous (debris, dustbin, animals, etc.), particularly those highlighted "
                "in rectangles. Pay special attention to traffic cones, barriers, and other highlighted objects. "
                "You must not discuss any objects beyond the seven categories above. For each identified object, provide actionable driving suggestions for the ego car, "
                "considering the object's position, status, and implications. Recommendations should prioritize safety, "
                "legal compliance, and efficient driving behavior."
            )
        prompts = [prompt_template.format(human_msg + ' Your comprehensive response should clear and accurate.')]
        inputs = processor(images=image, text=prompts, padding=True, return_tensors='pt').to(DEVICE, torch.float16)
        generation_config = {
            "max_new_tokens": MAX_LENGTH,
            "do_sample": False,
        }
        outputs = model.generate(**inputs, **generation_config)
        for sample_id, output in zip(sample_ids, outputs):
            generated_answer = processor.decode(output, skip_special_tokens=True).split('ASSISTANT: ')[1]
            print(generated_answer)
            predictions[sample_id] = generated_answer

        with open("submission_all.json", "w") as f:
            json.dump(predictions, f, indent=4)


