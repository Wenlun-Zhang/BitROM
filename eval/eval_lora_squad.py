import sys
sys.path.append("/home/zhang/Project/BitROM")
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.utils import replace_layer_with_lora, count_lora_parameters, compute_em, compute_f1, load_lora_parameters_from_state_dict
from safetensors.torch import load_file as load_safetensors


parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', type=str, default='/data2/user/zhang/LLM/Falcon3-7B-Instruct-1.58bit', help='Base LLM Model Directory')
parser.add_argument('--lora_dir', type=str, default='None', help='LoRA Model Directory')
parser.add_argument('--eval_base', action='store_true', default=False, help='Evaluate Base Model')
parser.add_argument('--eval_lora', action='store_true', default=False, help='Evaluate LoRA Model')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size in Evaluation')
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
# Data Preprocess Function
# Concatenate Context and Question as Prompt, and add Answer as hint
# Set token label in prompt to -100 for loss decoupling
# Add EOS token to answer to help model learning ending condition
# For unanswerable questions, set target as "No Answer" + eos_token
# =====================
def preprocess_function(example):
    prompt = "Context: " + example["context"].strip() + "\nQuestion: " + example["question"].strip() + "\nAnswer: "
    if not example["answers"]["text"]:
        target = "No Answer " + tokenizer.eos_token
    else:
        target = example["answers"]["text"][0].strip() + " " + tokenizer.eos_token
    full_text = prompt + target
    tokenized_full = tokenizer(full_text, truncation=True, max_length=512)
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=512)
    prompt_len = len(tokenized_prompt["input_ids"])
    labels = tokenized_full["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized_full["labels"] = labels
    tokenized_full["target"] = target
    tokenized_full["prompt"] = prompt
    return tokenized_full

# =====================
# Customize collate_fn Function
# =====================
def custom_collate_fn(examples):
    targets = [ex["target"] for ex in examples] if "target" in examples[0] else None
    prompts = [ex["prompt"] for ex in examples] if "prompt" in examples[0] else None
    examples_to_collate = [{k: v for k, v in ex.items() if k in ["input_ids", "attention_mask", "labels"]} for ex in examples]
    batch = tokenizer.pad([{k: ex[k] for k in ex if k != "labels"} for ex in examples_to_collate],
                          return_tensors="pt", padding=True)
    labels = [ex["labels"] for ex in examples_to_collate]
    max_len = max(len(l) for l in labels)
    padded_labels = [l + [-100] * (max_len - len(l)) for l in labels]
    batch["labels"] = torch.tensor(padded_labels)
    batch["targets"] = targets
    batch["prompts"] = prompts
    return batch

# =====================
# Load Validation Dataset and Tokenizer
# =====================
tokenizer = AutoTokenizer.from_pretrained(args.base_dir, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# =====================
# Load SQuAD1.1 Dataset (Train and Validation)
# =====================
dataset = load_dataset("squad")
cols_to_remove = [col for col in dataset["train"].column_names if col not in ["target", "prompt"]]
tokenized_val = dataset["validation"].map(preprocess_function, remove_columns=cols_to_remove)
val_dataloader = DataLoader(tokenized_val, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

# =====================
# Construct Evaluation Loop
# =====================
def clean_text(s):
    # Remove EOS Token
    return s.replace(tokenizer.eos_token, "").strip()


def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    total_em = 0
    total_f1 = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            prompts_list = batch["prompts"]
            inputs = tokenizer(prompts_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 32,
                early_stopping=True,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            for i, gen in enumerate(generated_ids):
                gen_text = tokenizer.decode(gen, skip_special_tokens=True)
                if "Answer: " in gen_text:
                    pred = gen_text.split("Answer:")[-1].strip()
                else:
                    pred = gen_text.strip()
                tgt = batch["targets"][i]
                tgt = clean_text(tgt)
                total_em += compute_em(pred, tgt)
                total_f1 += compute_f1(pred, tgt)
                total += 1
        avg_em = total_em / total if total > 0 else 0
        avg_f1 = total_f1 / total if total > 0 else 0
    return avg_em, avg_f1

# =====================
# Evaluate Models
# =====================
if args.eval_base:
    print("Start Evaluating Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.bfloat16, device_map={"": 0})
    base_model.to(device)
    import IPython; IPython.embed()
    base_em, base_f1 = evaluate(base_model, tokenizer, val_dataloader, device)
    print("Base Model Evaluation:")
    print("Exact Match (EM):", base_em)
    print("F1:", base_f1)

if args.eval_lora:
    print("Start Evaluating LoRA Model...")

    # Load Saved LoRA Model (safetensors)
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
    # Load LoRA Parameters from checkpoint

    state_dict = load_safetensors(os.path.join(args.lora_dir, "model.safetensors"))

    # import glob
    # shard_files = sorted(glob.glob(os.path.join(args.lora_dir, "model-*-of-*.safetensors")))
    # state_dict = {}
    # for shard in shard_files:
    #     shard_dict = load_safetensors(shard, device="cpu")
    #     state_dict.update(shard_dict)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    load_lora_parameters_from_state_dict(lora_model, state_dict, target_modules=target_modules)
    lora_em, lora_f1 = evaluate(lora_model, tokenizer, val_dataloader, device)
    print("Saved LoRA Model Evaluation:")
    print("Exact Match (EM):", lora_em)
    print("F1:", lora_f1)
