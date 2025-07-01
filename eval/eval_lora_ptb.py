import sys
sys.path.append("/home/zhang/Project/BitROM")
import os
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.utils import replace_layer_with_lora, count_lora_parameters, load_lora_parameters_from_state_dict
from safetensors.torch import load_file as load_safetensors

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', type=str, default='/data2/user/zhang/LLM/Falcon3-7B-Instruct-1.58bit', help='Base LLM Model Directory')
parser.add_argument('--lora_dir', type=str, default='None', help='LoRA Model Directory')
parser.add_argument('--eval_base', action='store_true', default=False, help='Evaluate Base Model')
parser.add_argument('--eval_lora', action='store_true', default=False, help='Evaluate LoRA Model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size in Evaluation')
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
# Load Tokenizer and Prepare Data for PTB
# =====================
tokenizer = AutoTokenizer.from_pretrained(args.base_dir, padding_side="left")
# Use eos_token as pad token to ensure pad_token_id is correct
tokenizer.pad_token = tokenizer.eos_token

# =====================
# Load PTB Dataset (Train and Validation)
# =====================
dataset = load_dataset("ptb_text_only", "penn_treebank")

# =====================
# Tokenization
# =====================
def tokenize_function(example):
    text = example.get("sentence", example.get("text"))
    return tokenizer(text)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
# Remove attention_mask for subsequent processing concentrate on input_ids
if "attention_mask" in tokenized_dataset["train"].column_names:
    tokenized_dataset = tokenized_dataset.remove_columns("attention_mask")

block_size = 512
def group_texts(examples):
    concatenated = []
    for ex in examples["input_ids"]:
        concatenated.extend(ex)
    total_length = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_length]
    result = {"input_ids": [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]}
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_val = tokenized_dataset["validation"].map(group_texts, batched=True)

# =====================
# Customize collate_fn Function for LM
# =====================
def lm_collate_fn(examples):
    # Use tokenizer.pad to pad input
    batch = tokenizer.pad(examples, return_tensors="pt")
    return batch

val_dataloader = DataLoader(tokenized_val, batch_size=args.batch_size, shuffle=False, collate_fn=lm_collate_fn)

# =====================
# Evaluation Function for Language Modeling (Perplexity)
# =====================
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss  # Average loss per token
            num_tokens = batch["input_ids"].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# =====================
# Evaluate Models
# =====================
if args.eval_base:
    print("Start Evaluating Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.bfloat16, device_map={"": 0})
    base_model.to(device)
    base_ppl = evaluate(base_model, val_dataloader, device)
    print("Base Model Perplexity:", base_ppl)

if args.eval_lora:
    print("Start Evaluating LoRA Model...")
    lora_model = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.bfloat16, device_map={"": 0})
    lora_model.to(device)
    target_modules = [name for name in
                      ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                      if getattr(args, name)]
    if args.quant_lora:
        replace_layer_with_lora(lora_model,
                                r=args.lora_rank,
                                alpha=args.lora_alpha,
                                target_modules=target_modules,
                                quant_lora=args.quant_lora,
                                x_bit=args.lora_x_bit,
                                lora_a_bit=args.lora_a_bit,
                                lora_b_bit=args.lora_b_bit)
    else:
        replace_layer_with_lora(lora_model,
                                r=args.lora_rank,
                                alpha=args.lora_alpha,
                                target_modules=target_modules)
    lora_params, total_params, lora_ratio = count_lora_parameters(lora_model)
    print(f"LoRA parameters: {lora_params}, Total parameters: {total_params}, Ratio: {lora_ratio:.2f}%")

    # Load LoRA parameters from checkpoint (safetensors)
    state_dict = load_safetensors(os.path.join(args.lora_dir, "model.safetensors"))
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    load_lora_parameters_from_state_dict(lora_model, state_dict, target_modules=target_modules)

    lora_ppl = evaluate(lora_model, val_dataloader, device)
    print("LoRA Model Perplexity:", lora_ppl)
