

MAX_LENGTH = 1000
USE_QLORA = True
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Download from HuggingFace
REPO_ID = ""
WANDB_PROJECT = "dlcv_project"
WANDB_NAME = "lora_and_seg"
OUTPUT_DIR = "model_all_seg"
DATA_ROOT = ""

import os
from tqdm import tqdm
import wandb
wandb.login()

from transformers import AutoProcessor
import torch.nn.functional as F

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
import torch.nn as nn
import lightning as L
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class CustomLlavaModel(LlavaForConditionalGeneration):
    """Custom LLaVA model that exposes intermediate features"""
    def __init__(self, config):
        super().__init__(config)
        #self._device = "cuda"
        
        # 初始化 Mask2Former 處理器
        self.seg_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-semantic"
        )
        
        # 延遲初始化 seg_model
        self.seg_model = None

        # Parameters for combining features
        self.seg_alpha = nn.Parameter(torch.zeros(1))
        
        # Project segmentation features to match LLaVA's feature dimension
        # Mask2Former's hidden size is 768, projecting to LLaVA's 4096
        self.seg_projection = nn.Linear(256, 4096)
        self.seg_projection.weight.data.normal_(mean=0.0, std=0.01)
        self.seg_projection.bias.data.zero_()

    def _init_seg_model(self):
        """延遲初始化 seg_model 並將其移動到當前設備"""
        if self.seg_model is None:
            self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-small-cityscapes-semantic",
                torch_dtype=torch.float16,
            ).to(self.device).eval()
    
    def get_seg_features(self, images):
        self._init_seg_model()
        """Extract seg features and reshape to match vision features"""
        batch_size = 1
        
        # 1. 獲取seg圖
        inputs = self.seg_processor(images=images, return_tensors="pt").to(self.seg_model.device)
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            seg_features = outputs.pixel_decoder_last_hidden_state  # [1, 256, 96, 96]
            
        
        
        # 3. 分割成 24x24 的 patches (得到 576 patches)
        # 分割成patches
        patch_size = 96 // 24  # = 4

        patches = seg_features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        # Average pool each patch
        patches = patches.mean(dim=(-2, -1))
        # Reshape to get patches
        patches = patches.permute(0, 2, 3, 1).reshape(1, 576, 256)

        
        # 4. 投影到與projection後的視覺特徵相同的維度 (4096)
        
        
        seg_features = self.seg_projection(patches)  # [B, 576, 4096]
        
        return seg_features
    
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
    
    def forward_language_model(self, inputs_embeds, attention_mask=None, labels=None):
        """Forward pass through the language model only"""
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
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
        combined_vision_features=None,  # 新增這個參數
        max_new_tokens=None,
        **kwargs
    ):
        if combined_vision_features is None:
            # 如果沒有提供組合特徵，就用原始的方式
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # 使用組合特徵的情況
        # 1. 獲取文字嵌入
        text_embeddings = self.get_text_embeddings(input_ids)
        
        # 2. 合併特徵
        merged_embeddings, merged_attention_mask, _ = self.merge_inputs_with_vision_features(
            combined_vision_features,  # 直接使用組合後的特徵
            text_embeddings,
            input_ids,
            attention_mask,
        )
        
        # 3. 通過語言模型生成
        outputs = self.language_model.generate(
            inputs_embeds=merged_embeddings,
            attention_mask=merged_attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return outputs

## Load model
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = CustomLlavaModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,

    #device_map="auto" 
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model","seg_projection","seg_model"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names: # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

for para in model.parameters(): 
    para.requires_grad = False

# 打開 seg 相關參數的訓練
model.seg_alpha.requires_grad = True
model.seg_projection.requires_grad = True

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
from PIL import Image
import os
from ultralytics import YOLO
import cv2
from PIL import Image, ImageEnhance

class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans = True):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans
        #self.bdd_model = YOLO("bdd_best.pt")
        #self.cone_model = YOLO("cone_best.pt")
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

from torch.utils.data import Subset
hf_dataset = {
    "train": load_dataset("ntudlcv/dlcv_2024_final1", split="train"),
    "val": load_dataset("ntudlcv/dlcv_2024_final1", split="val"),
    "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
}


custom_dataset = {
    "train": CodaDataset(hf_dataset["train"], exist_ans = True),
    "val": CodaDataset(hf_dataset["val"], exist_ans = True),
    "test": CodaDataset(hf_dataset["test"], exist_ans = False)
}
subset_indices = {
    "train": range(5),
    "val": range(5),
    "test": range(5)
}
subset_dataset = {
    "train": Subset(custom_dataset["train"], subset_indices["train"]),
    "val": Subset(custom_dataset["val"], subset_indices["val"]),
    "test": Subset(custom_dataset["test"], subset_indices["test"])
}

# %% [markdown]
# ### Define collate functions

# %%
def train_collate_fn(examples):
    images = []
    original_images = []
    texts = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        
        original_images.append(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "general":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, respones, and how they influence ego car. ASSISTANT: {assistant_msg}"
        elif task == "regional":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Please describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. ASSISTANT: {assistant_msg}"
        elif task == "suggestion":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene. ASSISTANT: {assistant_msg}"
        else:
            continue
        assert prompt != ""

        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    # labels = batch["input_ids"].clone()
    # labels[labels == processor.tokenizer.pad_token_id] = -100
    # batch["labels"] = labels
    labels = batch["input_ids"].clone()
    for i in range(len(texts)):
        # Find the position where assistant response starts
        input_ids = batch["input_ids"][i]
        attention_mask = batch["attention_mask"][i]
        
        # Mask everything before ASSISTANT: as -100
        assistant_pos = (input_ids == processor.tokenizer.convert_tokens_to_ids("ASSISTANT:")).nonzero()
        if len(assistant_pos) > 0:
            labels[i, :assistant_pos[-1] + 1] = -100  # Include the "ASSISTANT:" token in the mask
            
        # Also mask padding tokens
        labels[i][attention_mask == 0] = -100

    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, original_images, labels
def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    original_images = []
    texts = []
    answers = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        # Ensure image is in RGB mode
        original_images.append(image) 
        if image.mode != 'RGB':
            image = image.convert('RGB')

        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "general":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, respones, and how they influence ego car. ASSISTANT: "
        elif task == "regional":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Please describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. ASSISTANT: "
        elif task == "suggestion":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene. ASSISTANT: "
        else:
            continue
        assert prompt != ""

        texts.append(prompt)
        answers.append(assistant_msg)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, original_images, answers

# %% [markdown]
# ### Define PyTorch LightningModule

# %%
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np

class CustomLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, original_images, labels = batch
        
        # Print shapes and check for NaN values in inputs
        #print(f"\nInput Shapes:")
        #print(f"input_ids: {input_ids.shape}, range: [{input_ids.min()}, {input_ids.max()}]")
        #print(f"attention_mask: {attention_mask.shape}")
        #print(f"pixel_values: {pixel_values.shape}, range: [{pixel_values.min()}, {pixel_values.max()}]")
        #print(f"labels: {labels.shape}")
        
        # 1. Get vision features (Zv)
        vision_features = self.model.get_vision_features(pixel_values)
        print(f"Vision features: {vision_features.shape}, range: [{vision_features.min()}, {vision_features.max()}]")
        if torch.isnan(vision_features).any():
            print("Warning: NaN in vision features")
            
        # 2. Project vision features (Zv -> Hv)
        projected_vision_features = self.model.project_vision_features(vision_features)
        print(f"Projected features: {projected_vision_features.shape}, range: [{projected_vision_features.min()}, {projected_vision_features.max()}]")
        if torch.isnan(projected_vision_features).any():
            print("Warning: NaN in projected features")

        # 2.5 Get seg features, Combine features
        seg_features = self.model.get_seg_features(original_images)
        combined_features = projected_vision_features + self.model.seg_alpha * seg_features
            
        # 3. Get text embeddings (Xq)
        text_embeddings = self.model.get_text_embeddings(input_ids)
        print(f"Text embeddings: {text_embeddings.shape}, range: [{text_embeddings.min()}, {text_embeddings.max()}]")
        if torch.isnan(text_embeddings).any():
            print("Warning: NaN in text embeddings")
        
        # 4. Merge vision features with text embeddings
        merged_embeddings, merged_attention_mask, merged_labels = self.model.merge_inputs_with_vision_features(
            combined_features, 
            text_embeddings,
            input_ids,
            attention_mask,
            labels=labels
        )
        #print(f"Merged dimensions: embeddings={merged_embeddings.shape}, mask={merged_attention_mask.shape}, labels={merged_labels.shape if merged_labels is not None else None}")
        #print(f"Merged embeddings: {merged_embeddings.shape}, range: [{merged_embeddings.min()}, {merged_embeddings.max()}]")
        if torch.isnan(merged_embeddings).any():
            print("Warning: NaN in merged embeddings")
            
        # Ensure proper device placement
        merged_embeddings = merged_embeddings.to(self.device)
        merged_attention_mask = merged_attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Check if labels contain any non-masked positions
        valid_label_positions = (labels != -100).sum()
        print(f"Number of valid label positions: {valid_label_positions}")
        
        if valid_label_positions == 0:
            print("Warning: No valid label positions found!")
            return None
            
        # Apply gradient clipping at tensor level
        merged_embeddings = torch.clamp(merged_embeddings, min=-100, max=100)
        
        # 5. Forward through language model
        outputs = self.model.forward_language_model(
            inputs_embeds=merged_embeddings,
            attention_mask=merged_attention_mask,
            labels=merged_labels
        )
        
        loss = outputs.loss
        print(f"Loss: {loss.item()}")
        
        # Check if loss is NaN
        if torch.isnan(loss):
            print("Warning: Loss is NaN!")
            # You might want to skip this batch
            return None
            
        if loss.item() > 100:
            print("Warning: Loss is too large!")
            #loss = torch.clamp(loss, max=100)
        
        self.log("train_loss", loss)
        self.log("seg_alpha", self.model.seg_alpha)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, original_images, answers = batch
        
        try:
            # For validation, we use the model's generate method
            vision_features = self.model.get_vision_features(pixel_values)
            projected_vision_features = self.model.project_vision_features(vision_features)
            seg_features = self.model.get_seg_features(original_images)
            combined_features = projected_vision_features + self.model.seg_alpha * seg_features

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                combined_vision_features=combined_features,
                max_new_tokens=self.config.get("max_length", 1000),
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            predictions = self.processor.batch_decode(
                generated_ids[:, input_ids.size(1):],
                skip_special_tokens=True
            )
            print(predictions)
            
            # Compute validation metrics
            scores = []
            for pred, answer in zip(predictions, answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s>))", "", pred)
                score = edit_distance(pred, answer) / max(len(pred), len(answer))
                scores.append(score)
            
            avg_score = np.mean(scores)
            self.log("val_edit_distance", avg_score)
            return scores
            
        except Exception as e:
            print(f"Error in validation step: {str(e)}")
            return None
    
    def configure_optimizers(self):
        # Add weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get("lr"),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get("max_epochs"),
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def train_dataloader(self):
        return DataLoader(custom_dataset["train"], collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True)
            # custom_dataset
    def val_dataloader(self):
        return DataLoader(custom_dataset["val"], collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False)

# %%
config = {"max_epochs": 20,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0, #0.5,
          "accumulate_grad_batches": 8,#2,
          "lr": 1e-4,#1e-5,
          "batch_size": 1,
          "verbose": True,
          "max_length": 1000#,
          #"warmup_steps": 100
}

model_module = CustomLlavaModelPLModule(config, processor, model)


# %% [markdown]
# ### Train

# %%
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        output_dir = f"{OUTPUT_DIR}/general-epoch-{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        pl_module.processor.save_pretrained(output_dir)
        seg_state = {
            'seg_alpha': pl_module.model.seg_alpha.data,
            'seg_projection.weight': pl_module.model.seg_projection.weight.data,
            'seg_projection.bias': pl_module.model.seg_projection.bias.data
        }
        torch.save(seg_state, os.path.join(output_dir, 'seg_params.bin'))
        #print(f"Saved seg_alpha value: {pl_module.model.seg_alpha.item()}")



early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=10, verbose=False, mode="min")

# %%
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)
ddp_strategy = DDPStrategy(find_unused_parameters=True)
trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        strategy=ddp_strategy,
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback()]
)

#trainer.fit(model_module)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"參數名稱: {name},")
model.seg_alpha.requires_grad = True
model.seg_projection.requires_grad = True
print(f"seg_alpha requires_grad: {model.seg_alpha.requires_grad}")
print(f"seg_alpha value: {model.seg_alpha.item()}")
trainer.fit(model_module)