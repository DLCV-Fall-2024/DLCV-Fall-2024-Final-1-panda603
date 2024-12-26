# %%
MAX_LENGTH = 800
USE_QLORA = True
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Download from HuggingFace
REPO_ID = ""
WANDB_PROJECT = "dlcv_project"
WANDB_NAME = "regional"
OUTPUT_DIR = "model"
DATA_ROOT = ""
from tqdm import tqdm
import os
# import multiprocessing as mp
# try:
#     mp.set_start_method('spawn', force=True)
#     print("spawned")
# except RuntimeError:
#     pass
# %%
import wandb
# os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb.login()

# %% [markdown]
# ### Load processor

# %%
from transformers import AutoProcessor
from PIL import Image, ImageEnhance
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

# %% [markdown]
# ### Load model

# %%
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

## Load model
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )
else:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )


# %% [markdown]
# ### Apply PEFT

# %%
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
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# %% [markdown]
# ### Create PyTorch dataset

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
from PIL import Image
from itertools import islice
import os
import cv2
class CodaDataset(Dataset):
    def __init__(self, hf_dataset, exist_ans = True):
        self.hf_dataset = hf_dataset
        self.exist_ans = exist_ans

        self.valid_indices = []
        self.filtered_indices = [
            i for i in tqdm(range(len(hf_dataset)))
            if hf_dataset[i]["id"].split("_")[1] in ["regional"]
        ]
    def process_image(self, image):

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
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        
        # sample = self.hf_dataset[idx]
        actual_idx = self.filtered_indices[idx]
        sample = self.hf_dataset[actual_idx]
        _, task, _ = sample["id"].split("_")
        original_image = sample["image"]
        processed_image = self.process_image(original_image)
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
    # "test": load_dataset("ntudlcv/dlcv_2024_final1", split="test")
}


custom_dataset = {
    "train": CodaDataset(hf_dataset["train"], exist_ans = True),
    "val": CodaDataset(hf_dataset["val"], exist_ans = True),
    # "test": CodaDataset(hf_dataset["test"], exist_ans = False)
}
subset_indices = {
    "train": range(5),  # First 2 samples
    "val": range(5),
    # "test": range(5)
}
subset_dataset = {
    "train": Subset(custom_dataset["train"], subset_indices["train"]),
    "val": Subset(custom_dataset["val"], subset_indices["val"]),
    # "test": Subset(custom_dataset["test"], subset_indices["test"])
}

# %% [markdown]
# ### Define collate functions

# %%
def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        sample_id, task, image, human_msg, assistant_msg = example
        if task not in ["regional"]:
            continue
        # print(sample_id, task, image, human_msg, assistant_msg)
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "regional":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. ASSISTANT: {assistant_msg}"
        else:
            continue
        assert prompt != ""

        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

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
        sample_id, task, image, human_msg, assistant_msg = example
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        images.append(image)
        human_msg = human_msg.split('<image>\n', 1)[-1]
        
        if task == "regional":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. ASSISTANT:"
        else:
            continue
        assert prompt != ""

        texts.append(prompt)
        answers.append(assistant_msg)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers

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

        input_ids, attention_mask, pixel_values, labels = batch


        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            use_cache = True,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don"t skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
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
        return DataLoader(custom_dataset["train"], collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=8)
            # custom_dataset
    def val_dataloader(self):
        return DataLoader(custom_dataset["val"], collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=8)

# %%
config = {"max_epochs": 20,
        #   "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          # "seed":2022,
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)


# %% [markdown]
# ### Train

# %%
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from huggingface_hub import HfApi

class LitProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True
        self.prediction_bar = True
    
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items["loss"] = f"{trainer.callback_metrics['train_loss']:.3f}"
        return items


api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        output_dir = f"{OUTPUT_DIR}/llava-fintune-epoch-regional{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        pl_module.processor.save_pretrained(output_dir)
        # pl_module.model.push_to_hub(REPO_ID,
        #                             commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    # def on_train_end(self, trainer, pl_module):
    #     print(f"Pushing model to the hub after training")
    #     pl_module.processor.push_to_hub(REPO_ID,
    #                                 commit_message=f"Training done")
    #     pl_module.model.push_to_hub(REPO_ID,
    #                                 commit_message=f"Training done")

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
        callbacks=[
            PushToHubCallback(),
            LitProgressBar()
        ],
        enable_progress_bar=True  
)

#trainer.fit(model_module)
trainer.fit(model_module)