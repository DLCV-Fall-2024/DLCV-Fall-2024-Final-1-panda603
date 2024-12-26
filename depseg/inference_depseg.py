from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
from PIL import Image, ImageEnhance
import numpy as np
from ultralytics import YOLO
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import PeftModel

# Constants
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER_PATH = "model_all/general-epoch-0"  # Update with your model path
MAX_LENGTH = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load RAM predictions if needed
with open('tag_prediction.json') as f:
    ram_pred = json.load(f)

class CustomLlavaModel(LlavaForConditionalGeneration):
    """Custom LLaVA model with depth features"""
    def __init__(self, config):
        super().__init__(config)
        self.depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

        self.depth_alpha = nn.Parameter(torch.zeros(1))
        self.depth_projection = nn.Linear(1, 4096)
        self.depth_projection.weight.data.normal_(mean=0.0, std=0.01)
        self.depth_projection.bias.data.zero_()
    
    def get_depth_features(self, images):
        """Extract depth features and reshape to match vision features"""
        batch_size = 1
        
        inputs = self.depth_processor(images=images, return_tensors="pt").to(self.depth_model.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            
        depth_map = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(336, 336),
            mode="bicubic"
        )
        depth_map = depth_map.squeeze(1)
        
        patch_size = 336 // 24  # = 14
        patches = depth_map.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.mean(dim=(-2, -1))
        patches = patches.reshape(batch_size, 576, 1)
        
        depth_features = self.depth_projection(patches)
        
        return depth_features
    def get_vision_features(self, pixel_values):
        """Extract vision features (Zv) from the vision encoder"""
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=False)
        selected_vision_feature = vision_outputs.last_hidden_state
        
        if self.config.vision_feature_select_strategy == "default":
            selected_vision_feature = selected_vision_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_vision_feature = selected_vision_feature
            
        return selected_vision_feature  # This is Zv
    def project_vision_features(self, vision_features):
        """Project vision features (Zv -> Hv) through the multi-modal projector"""
        return self.multi_modal_projector(vision_features)  # This is Hv
    
    def get_text_embeddings(self, input_ids):
        """Get text embeddings from the language model"""
        return self.language_model.model.embed_tokens(input_ids)  # This is Xq
    def merge_inputs_with_vision_features(self, vision_features, text_embeddings, input_ids, attention_mask, labels=None):
        batch_size, sequence_length = input_ids.shape
        num_images, num_patches, embed_dim = vision_features.shape
        
        #print(f"\nMerging details:")
        #print(f"Initial sequence length: {sequence_length}")
        #print(f"Number of patches: {num_patches}")
        
        # Initialize lists to store segments
        final_segments = []
        attention_segments = []
        labels_segments = [] if labels is not None else None
        
        # Process each batch item
        for b in range(batch_size):
            # Find the image token position
            image_token_mask = input_ids[b] == self.config.image_token_index
            image_pos = torch.where(image_token_mask)[0][0]
            
            #print(f"Image token position: {image_pos}")
            #print(f"Text before image: {image_pos} tokens")
            #print(f"Text after image: {sequence_length - image_pos - 1} tokens")
            
            # 1. Add text before image token
            if image_pos > 0:
                final_segments.append(text_embeddings[b, :image_pos])
                attention_segments.append(attention_mask[b, :image_pos])
                if labels is not None:
                    labels_segments.append(labels[b, :image_pos])
            
            # 2. Add image features
            final_segments.append(vision_features[0])
            attention_segments.append(torch.ones(num_patches, device=attention_mask.device))
            if labels is not None:
                labels_segments.append(torch.full((num_patches,), -100, device=labels.device, dtype=labels.dtype))
            
            # 3. Add remaining text
            if image_pos < sequence_length - 1:
                final_segments.append(text_embeddings[b, image_pos+1:])
                attention_segments.append(attention_mask[b, image_pos+1:])
                if labels is not None:
                    labels_segments.append(labels[b, image_pos+1:])
        
        # Concatenate all segments
        final_embedding = torch.cat(final_segments, dim=0)
        final_attention_mask = torch.cat(attention_segments, dim=0)
        final_labels = None
        if labels is not None:
            final_labels = torch.cat(labels_segments, dim=0)
        
        #print(f"Final sequence length: {final_embedding.size(0)}")
        if labels is not None:
            print(f"Labels sequence length: {final_labels.size(0)}")
        
        return (final_embedding.unsqueeze(0), 
                final_attention_mask.unsqueeze(0),
                final_labels.unsqueeze(0) if final_labels is not None else None)
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        combined_vision_features=None,
        max_new_tokens=None,
        **kwargs
    ):
        if combined_vision_features is None:
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        text_embeddings = self.get_text_embeddings(input_ids)
        text_embeddings = text_embeddings.to(combined_vision_features.dtype)
        merged_embeddings, merged_attention_mask, _ = self.merge_inputs_with_vision_features(
            combined_vision_features,
            text_embeddings,
            input_ids,
            attention_mask,
        )

        with torch.autocast("cuda"):
            outputs = self.language_model.generate(
                inputs_embeds=merged_embeddings,
                attention_mask=merged_attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        return outputs

class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans=False):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans
        #self.bdd_model = YOLO("bdd_best.pt")
        #self.cone_model = YOLO("cone_best.pt")

    def process_image_with_yolo(self, image):
        image_np = np.array(image)
        bdd_results = self.bdd_model(image_np)[0]
        cone_results = self.cone_model(image_np)[0]
        annotated_image = image_np.copy()
        
        colors = {
            'bdd100k': (0, 255, 0),
            'traffic_cone': (0, 0, 255)
        }
        
        for result in bdd_results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         colors['bdd100k'], 
                         2)
        
        for result in cone_results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         colors['traffic_cone'], 
                         2)
        
        return Image.fromarray(annotated_image)

    def brighten_image(self, image):
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        contrast_enhanced = contrast_enhancer.enhance(1.2)
        
        contrast_img_cv2 = cv2.cvtColor(np.array(contrast_enhanced), cv2.COLOR_RGB2BGR)
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_img_cv2, -1, kernel)
        
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        
        # sample = self.hf_dataset[idx]
        sample = self.hf_dataset[idx]
        _, task, _ = sample["id"].split("_")
        original_image = sample["image"]

        if task in ["yolo"]:
            processed_image = self.process_image_with_yolo(original_image)
        else:
            processed_image = self.brighten_image(original_image)
            width, height = processed_image.size #要改
            max_dim = max(width, height)
            image = processed_image.resize((max_dim, max_dim))
        
        conversation = sample["conversations"]
        if self.exist_ans:
            human_msg = conversation[0]["value"]
            assistant_msg = conversation[1]["value"]
            return sample["id"], task, image, human_msg, assistant_msg #要改
        else:
            human_msg = conversation[0]["value"]
            return sample["id"], task, image, human_msg #要改

def main():
    # Initialize model and processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = CustomLlavaModel.from_pretrained(
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
    depth_state = torch.load(f"{ADAPTER_PATH}/depth_params.bin", map_location=DEVICE)
    model.model.depth_alpha = nn.Parameter(depth_state['depth_alpha'])
    model.model.depth_projection.weight.data = depth_state['depth_projection.weight']
    model.model.depth_projection.bias.data = depth_state['depth_projection.bias']
    model.eval()

    # Load dataset
    hf_dataset = {
        "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
    }
    custom_dataset = {
        "test": CodaDataset(hf_dataset["test"], exist_ans=False)
    }

    def collate_fn(batch):
        return zip(*batch)

    test_loader = DataLoader(
        custom_dataset["test"],
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    prompt_template = 'USER: {} ASSISTANT:'
    predictions = {}

    with torch.no_grad():
        for sample_ids, tasks, images, human_msgs in tqdm(test_loader):
            sample_id = sample_ids[0]
            task = tasks[0]
            image = images[0]
            
            if task == "general":
                human_msg = "<image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, respones, and how they influence ego car."
            elif task == "regional":
                human_msg = "<image>\n Please describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving."
            elif task == "suggestion":
                human_msg = "<image>\n There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene."

            prompts = [prompt_template.format(human_msg + ' Your comprehensive response should clear and accurate.')]
            inputs = processor(images=image, text=prompts, padding=True, return_tensors='pt').to(DEVICE, torch.float16)
            
            # Get vision and depth features
            vision_features = model.model.get_vision_features(inputs.pixel_values)
            projected_vision_features = model.model.project_vision_features(vision_features)
            depth_features = model.model.get_depth_features(image)
            
            # Combine features using learned alpha
            combined_features = projected_vision_features + model.model.depth_alpha * depth_features
            print("123:")
            print(model.model.depth_alpha)

            # Generate response
            outputs = model.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                combined_vision_features=combined_features,
                max_new_tokens=MAX_LENGTH,
                do_sample=False,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            try:
                generated_answer = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[1]
            except:
                generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
                print("87?")
            predictions[sample_id] = generated_answer
            
            print(f"\nTask: {task}, Sample ID: {sample_id}")
            print(f"Generated answer: {generated_answer}\n")

        # Save predictions
        with open("submission_depth.json", "w") as f:
            json.dump(predictions, f, indent=4)

if __name__ == "__main__":
    main()