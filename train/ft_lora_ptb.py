import sys
sys.path.append("/home/zhang/Project/BitROM")
import gc
import os
import argparse
import math
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.utils import replace_layer_with_lora, count_lora_parameters
from module.lora import LoRALinear, QuantLoRALinear

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='/data2/user/zhang/LLM/Falcon3-7B-Instruct-1.58bit', help='LLM Model Directory')
parser.add_argument('--epoch', type=int, default=3, help='Training Epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size in Finetune')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning Rate')
parser.add_argument('--save_dir', type=str, default='./lora_ft_model/ptb/Falcon3_158_7B_16r_vod_w6a8', help='Save Output LoRA Model Directory')
parser.add_argument('--q_proj', action='store_true', default=False, help='LoRA at Q Projection')
parser.add_argument('--k_proj', action='store_true', default=False, help='LoRA at K Projection')
parser.add_argument('--v_proj', action='store_true', default=True, help='LoRA at V Projection')
parser.add_argument('--o_proj', action='store_true', default=True, help='LoRA at O Projection')
parser.add_argument('--gate_proj', action='store_true', default=False, help='LoRA at Gate Projection')
parser.add_argument('--up_proj', action='store_true', default=False, help='LoRA at Up Projection')
parser.add_argument('--down_proj', action='store_true', default=True, help='LoRA at Down Projection')
parser.add_argument('--lora_rank', type=int, default=16, help='LoRA Rank Parameter r')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA Scale Parameter alpha')
parser.add_argument('--quant_lora', action='store_true', default=False, help='LoRA Adapter Quantization')
parser.add_argument('--lora_x_bit', type=int, default=8, help='LoRA Activation Bit Width')
parser.add_argument('--lora_a_bit', type=int, default=6, help='LoRA A Bit Width')
parser.add_argument('--lora_b_bit', type=int, default=6, help='LoRA B Bit Width')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Load Model and Tokenizer
# =====================
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}  # Confine to GPU0
)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
# Use eos_token as pad token to ensure pad_token_id is correct
tokenizer.pad_token = tokenizer.eos_token

# =====================
# Replace LoRA Modules and Calculate LoRA Parameters
# =====================
target_modules = [name for name in
                  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                  if getattr(args, name)]
if args.quant_lora:
    replace_layer_with_lora(model,
                            r=args.lora_rank,
                            alpha=args.lora_alpha,
                            target_modules=target_modules,
                            quant_lora=args.quant_lora,
                            x_bit=args.lora_x_bit,
                            lora_a_bit=args.lora_a_bit,
                            lora_b_bit=args.lora_b_bit)
else:
    replace_layer_with_lora(model,
                            r=args.lora_rank,
                            alpha=args.lora_alpha,
                            target_modules=target_modules)

lora_params, total_params, lora_ratio = count_lora_parameters(model)
print(f"LoRA parameters: {lora_params}, Total parameters: {total_params}, Ratio: {lora_ratio:.2f}%")

# Freeze all parameters and enable gradients only for LoRA modules
for param in model.parameters():
    param.requires_grad = False
for name, module in model.named_modules():
    if isinstance(module, (LoRALinear, QuantLoRALinear)):
        module.lora_a.requires_grad = True
        module.lora_b.requires_grad = True

# =====================
# Load PTB Dataset (Train and Validation)
# =====================
dataset = load_dataset("ptb_text_only", "penn_treebank")

# =====================
# Data Preprocessing for Language Modeling
# =====================
def tokenize_function(example):
    text = example.get("text", example.get("sentence"))
    return tokenizer(text)

# Tokenize dataset and remove original text
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
# Remove attention_mask for subsequent processing concentrate on input_ids
if "attention_mask" in tokenized_dataset["train"].column_names:
    tokenized_dataset = tokenized_dataset.remove_columns("attention_mask")

block_size = 512
def group_texts(examples):
    # Process input_ids
    concatenated = []
    for ex in examples["input_ids"]:
        concatenated.extend(ex)
    total_length = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_length]
    result = {"input_ids": [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]}
    # Labels equal to input_ids in aoto-regressive training
    result["labels"] = result["input_ids"].copy()
    return result

lm_train = tokenized_dataset["train"].map(group_texts, batched=True)
lm_val = tokenized_dataset["validation"].map(group_texts, batched=True)

# =====================
# Customize collate_fn Function for LM
# =====================
def lm_collate_fn(examples):
    batch = tokenizer.pad(examples, return_tensors="pt")
    return batch

# =====================
# Construct Data Loader
# =====================
train_dataloader = DataLoader(lm_train, batch_size=args.batch_size, shuffle=True, collate_fn=lm_collate_fn)
val_dataloader = DataLoader(lm_val, batch_size=args.batch_size, shuffle=False, collate_fn=lm_collate_fn)

# =====================
# Training Settings
# =====================
model.to(device)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
num_training_steps = args.epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# =====================
# Training Loop
# =====================
model.train()
global_step = 0
for epoch in range(args.epoch):
    print(f"Epoch {epoch + 1}/{args.epoch}")
    epoch_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        global_step += 1

        if global_step % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item()}")

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss}")

    # =====================
    # Evaluation: Compute Perplexity on Validation Set
    # =====================
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
    avg_loss_val = total_loss / total_tokens
    perplexity = math.exp(avg_loss_val)
    print(f"Validation Perplexity: {perplexity}")

    model.train()
    torch.cuda.empty_cache()
    gc.collect()

# =====================
# Save LoRA Model
# =====================
os.makedirs(args.save_dir, exist_ok=True)
model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print("Finetuned LoRA Model Save To: ", args.save_dir)
