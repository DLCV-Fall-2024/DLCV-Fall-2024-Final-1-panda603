# %%
MAX_LENGTH = 800
USE_QLORA = True
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Download from HuggingFace
REPO_ID = ""
WANDB_PROJECT = "dlcv_project"
WANDB_NAME = "all_task_multilora32_eos"
OUTPUT_DIR = "model_all_task_multilora32_eos"
DATA_ROOT = ""

import os
from tqdm import tqdm
import wandb
# wandb.login()

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

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
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

TASK_NAMES = ["general", "regional", "suggestion"]
for task in TASK_NAMES:
    model.add_adapter(task, lora_config)
# model.add_adapter("vision", lora_config_vision)
# model.set_adapter("general")
# model.set_adapter("vision")
print(model.active_adapters)

# assert False
print(model)

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
        self.bdd_model = YOLO("bdd_best.pt")
        self.cone_model = YOLO("cone_best.pt")
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
    texts = []
    tasks = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        tasks.append(task)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "general":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, "
            "specializing in recognizing traffic scenes and making detailed explanations. "
            "The assistant receives an image of traffic captured from the perspective of the ego car. "
            "USER: <image>\n Focus on objects in the image that directly influence the ego car's driving behavior: "
            "vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, "
            "motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, "
            "yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.), particularly those "
            "highlighted in rectangles. Pay special attention to traffic cones, barriers, and other highlighted "
            "objects. You must not discuss any objects beyond the seven categories above. For each highlighted "
            "object, describe its color, position, status, implications for driving, recommended responses, and "
            f"its influence on the ego car's behavior. ASSISTANT: {assistant_msg}"
            )
        elif task == "regional":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, specializing in recognizing "
            "traffic scenes and making detailed explanation. The assistant receives an image of traffic captured "
            "from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red "
            "rectangle in the image. Describe its color, position, status, implication, response, and explain why "
            f"it affect ego car driving. ASSISTANT: {assistant_msg}"
            )
        elif task == "suggestion":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, specializing in recognizing "
            "traffic scenes and making detailed explanations. The assistant receives an image of traffic captured "
            "from the perspective of the ego car. USER: <image>\n Focus on objects in the image that directly "
            "influence the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), "
            "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, "
            "directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, "
            "dustbin, animals, etc.), particularly those highlighted in rectangles. Pay special attention to traffic "
            "cones, barriers, and other highlighted objects. You must not discuss any objects beyond the seven "
            "categories above. For each identified object, provide actionable driving suggestions for the ego car, "
            "considering the object's position, status, and implications. Recommendations should prioritize safety, "
            f"legal compliance, and efficient driving behavior. ASSISTANT: {assistant_msg}"
            )
        else:
            continue
        assert prompt != ""

        texts.append(prompt + '</s>')

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

    return input_ids, attention_mask, pixel_values, labels, tasks
def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    tasks = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        tasks.append(task)
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "general":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, "
            "specializing in recognizing traffic scenes and making detailed explanations. "
            "The assistant receives an image of traffic captured from the perspective of the ego car. "
            "USER: <image>\n Focus on objects in the image that directly influence the ego car's driving behavior: "
            "vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, "
            "motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, "
            "yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.), particularly those "
            "highlighted in rectangles. Pay special attention to traffic cones, barriers, and other highlighted "
            "objects. You must not discuss any objects beyond the seven categories above. For each highlighted "
            "object, describe its color, position, status, implications for driving, recommended responses, and "
            "its influence on the ego car's behavior. ASSISTANT: "
            )
        elif task == "regional":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, specializing in recognizing "
            "traffic scenes and making detailed explanation. The assistant receives an image of traffic captured "
            "from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red "
            "rectangle in the image. Describe its color, position, status, implication, response, and explain why "
            "it affect ego car driving. ASSISTANT: "
            )
        elif task == "suggestion":
            prompt = (
            "A chat between a curious human and an autonomous driving assistant, specializing in recognizing "
            "traffic scenes and making detailed explanations. The assistant receives an image of traffic captured "
            "from the perspective of the ego car. USER: <image>\n Focus on objects in the image that directly "
            "influence the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), "
            "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, "
            "directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, "
            "dustbin, animals, etc.), particularly those highlighted in rectangles. Pay special attention to traffic "
            "cones, barriers, and other highlighted objects. You must not discuss any objects beyond the seven "
            "categories above. For each identified object, provide actionable driving suggestions for the ego car, "
            "considering the object's position, status, and implications. Recommendations should prioritize safety, "
            "legal compliance, and efficient driving behavior. ASSISTANT: "
            )
        else:
            continue
        assert prompt != ""

        texts.append(prompt)
        answers.append(assistant_msg)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers, tasks

# %% [markdown]
# ### Define PyTorch LightningModule

# %%
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

        input_ids, attention_mask, pixel_values, labels, tasks = batch
        # Get unique tasks in the batch
        unique_tasks = list(set(tasks))
        total_loss = 0

        for task in unique_tasks:
            task_indices = [i for i, t in enumerate(tasks) if t == task]
            task_input_ids = input_ids[task_indices]
            task_attention_mask = attention_mask[task_indices]
            task_pixel_values = pixel_values[task_indices]
            task_labels = labels[task_indices]

            # Set the active adapter
            self.model.set_adapter(task)

            outputs = self.model(input_ids=task_input_ids,
                                attention_mask=task_attention_mask,
                                pixel_values=task_pixel_values,
                                labels=task_labels,
                                use_cache=True
                            )

            loss = outputs.loss
            total_loss += loss

            # Log the loss for each task
            self.log(f"train_loss_{task}", loss, prog_bar=True)

        total_loss /= len(unique_tasks)
        self.log("train_loss", total_loss)


        return total_loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers, tasks = batch
        unique_tasks = list(set(tasks))
        
        # autoregressively generate token IDs
        scores = []
        for task in unique_tasks:
            task_indices = [i for i, t in enumerate(tasks) if t == task]
            task_input_ids = input_ids[task_indices]
            task_attention_mask = attention_mask[task_indices]
            task_pixel_values = pixel_values[task_indices]
            task_answers = [answers[x] for x in task_indices] # answers[task_indices]

            # Set the active adapter
            self.model.set_adapter(task)

            generated_ids = self.model.generate(input_ids=task_input_ids, attention_mask=task_attention_mask,
                                       pixel_values=task_pixel_values, max_new_tokens=MAX_LENGTH)

        
            # generated_ids = torch.stack(generated_ids)
            # turn them back into text, chopping of the prompt
            # important: we don"t skip special tokens here, because we want to see them in the output
            predictions = self.processor.batch_decode(generated_ids[:, task_input_ids.size(1):], skip_special_tokens=True)

            for pred, answer in zip(predictions, task_answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

                if self.config.get("verbose", False) and len(scores) == 1:
                    print(f"Prediction: {pred}")
                    print(f"    Answer: {answer}")
                    print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(custom_dataset["train"], collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True)
            # custom_dataset
    def val_dataloader(self):
        return DataLoader(custom_dataset["val"], collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False)

# %%
config = {"max_epochs": 20,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)


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

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=10, verbose=False, mode="min")

# %%
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

os.environ["WANDB__SERVICE_WAIT"] = "300"
# wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)
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
        # logger=wandb_logger,
        callbacks=[PushToHubCallback()]
)

trainer.fit(model_module)