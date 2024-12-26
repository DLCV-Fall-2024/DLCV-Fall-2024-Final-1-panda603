MAX_LENGTH = 1000
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Download from HuggingFace
REPO_ID = ""
WANDB_PROJECT = "dlcv_project"
WANDB_NAME = "test"
OUTPUT_DIR = "model"
DATA_ROOT = ""
import json

import os
from tqdm import tqdm
import wandb
wandb.login()
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

## Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="cuda:0"
)
# Apply PEFT
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
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
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

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
class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans = True):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans
        self.bdd_model = YOLO("bdd_best.pt")
        self.cone_model = YOLO("cone_best.pt")
        self.filtered_indices = [
            i for i in tqdm(range(len(hf_dataset)))
            if (hf_dataset[i]["id"].split("_")[1] in ["general"] and 
                int(hf_dataset[i]["id"].split("_")[-1]) < 3000)
        ]
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
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        
        # sample = self.hf_dataset[idx]
        actual_idx = self.filtered_indices[idx]
        sample = self.hf_dataset[actual_idx]
        _, task, _ = sample["id"].split("_")
        original_image = sample["image"]
        processed_image = self.process_image_with_yolo(original_image)
        conversation = sample["conversations"]
        if self.exist_ans:
            human_msg = conversation[0]["value"]
            assistant_msg = conversation[1]["value"]
            return sample["id"], task, processed_image, human_msg, assistant_msg
        else:
            human_msg = conversation[0]["value"]
            return sample["id"], task, processed_image, human_msg

from torch.utils.data import Subset
hf_dataset = {
    "train": load_dataset("ntudlcv/dlcv_2024_final1", split="train"),
    # "val": load_dataset("ntudlcv/dlcv_2024_final1", split="val"),
    "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
}


custom_dataset = {
    "train": CodaDataset(hf_dataset["train"], exist_ans = True),
    # "val": CodaDataset(hf_dataset["val"], exist_ans = True),
    "test": CodaDataset(hf_dataset["test"], exist_ans = False)
}
subset_indices = {
    "train": range(5),
    # "val": range(5),
    "test": range(5)
}
subset_dataset = {
    "train": Subset(custom_dataset["train"], subset_indices["train"]),
    # "val": Subset(custom_dataset["val"], subset_indices["val"]),
    "test": Subset(custom_dataset["test"], subset_indices["test"])
}
with open("general_training_v2.json", "r") as f:
    full_data = json.load(f)
    first_stage = {k: v for k, v in full_data.items() if int(k.split("_")[-1]) < 3000}
with open("submission_first.json", "r") as f:
    full_data = json.load(f)
    test_stage = {k: v for k, v in full_data.items() if str(k.split("_")[1]) == "general"}

def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        if task != "general":
            continue
        prompt = (
            "A chat between a human and an autonomous driving assistant that specializes in analyzing traffic scenes."
            f"USER: <image>\n INITIAL DESCRIPTION: {first_stage[sample_id]}"
            "REVIEW TASK: Examine the image again and verify the initial description. If accurate, keep it unchanged. "
            "If any mistakes are found regarding vehicles, vulnerable road users, traffic signs, traffic lights, traffic cones, barriers, "
            "or miscellaneous safety objects, directly provide a complete refined description of the scene focusing on these elements and their influence on the ego "
            f"car's driving behavior. ASSISTANT: {assistant_msg}"
        )
        # prompt = (
        #     "A chat between a curious human and an autonomous driving assistant, specializing in recognizing traffic scenes "
        #     "and providing detailed explanations. The assistant receives an image of traffic captured from the perspective "
        #     "of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, "
        #     "trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic "
        #     "signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, "
        #     "miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories "
        #     "above. Please describe each object's color, position, status, implication, respones, and how they influence "
        #     f"ego car. ASSISTANT (Draft Version): {first_stage[sample_id]}\n Review Task: This is the draft version of the "
        #     f"inference generated by our system. Please "
        #     "carefully review the image again, focusing on the seven categories mentioned earlier (vehicles, vulnerable road "
        #     "users, traffic signs, traffic lights, traffic cones, barriers, and miscellaneous). Identify any inaccuracies, "
        #     "missing details, or incorrect interpretations in the draft version. Provide a corrected and comprehensive final "
        #     f"answer that accurately describes the objects and their influence on the ego car's driving behavior. ASSISTANT: "
        #     f"{assistant_msg}"
        # )
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

    return input_ids, attention_mask, pixel_values, labels
def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        sample_id, task, image, human_msg = example
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        
        if task != "general":
            continue
        # suggest_key = "Test_general_"+str(sample_id.split("_")[2])
        prompt = (
            "A chat between a human and an autonomous driving assistant that specializes in analyzing traffic scenes."
            f"USER: <image>\n INITIAL DESCRIPTION: {test_stage[sample_id]}"
            "REVIEW TASK: Examine the image again and verify the initial description. If accurate, keep it unchanged. "
            "If any mistakes are found regarding vehicles, vulnerable road users, traffic signs, traffic lights, traffic cones, barriers, "
            "or miscellaneous safety objects, directly provide a complete refined description of the scene focusing on these elements and their influence on the ego "
            f"car's driving behavior. ASSISTANT: "
        )
        # elif task == "suggestion":
        #     prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanations. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects in the image that directly influence the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.),  particularly those highlighted in rectangles. Pay special attention to traffic cones, barriers, and other highlighted objects. You must not discuss any objects beyond the seven categories above. For each identified object, provide actionable driving suggestions for the ego car, considering the object's position, status, and implications. Recommendations should prioritize safety, legal compliance, and efficient driving behavior. ASSISTANT: {assistant_msg}"
        assert prompt != ""

        texts.append(prompt)
        # answers.append(assistant_msg)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values

import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np

class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")
    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, labels = batch


        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            use_cache = True,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return {"loss": loss} 

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values = batch
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don"t skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(predictions)
    #     scores = []
    #     for pred, answer in zip(predictions, answers):
    #         pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    #         scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

    #         if self.config.get("verbose", False) and len(scores) == 1:
    #             print(f"Prediction: {pred}")
    #             print(f"    Answer: {answer}")
    #             print(f" Normed ED: {scores[0]}")

    #     self.log("val_edit_distance", np.mean(scores))

    #     return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(custom_dataset["train"], collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=0)
    def val_dataloader(self):
        return DataLoader(custom_dataset["test"], collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=0)

config = {"max_epochs": 20,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
}

model_module = LlavaModelPLModule(config, processor, model)

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        output_dir = f"{OUTPUT_DIR}/reviewer-epoch-{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        pl_module.processor.save_pretrained(output_dir)

# early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=10, verbose=False, mode="min")
class BestModelCallback(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.best_loss = float('inf')
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_loss = outputs['loss'].item()
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            print(f"New best loss: {self.best_loss:.4f}. Saving model...")
            output_path = f"{self.output_dir}/best_model"
            pl_module.model.save_pretrained(output_path)
            pl_module.processor.save_pretrained(output_path)
best_model_callback = BestModelCallback(OUTPUT_DIR)

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
        logger=wandb_logger,
        callbacks=[PushToHubCallback(), best_model_callback]
)
trainer.fit(model_module)