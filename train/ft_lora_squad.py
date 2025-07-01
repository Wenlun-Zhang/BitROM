import sys
sys.path.append("/home/zhang/Project/BitROM")
import gc
import os
import argparse
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.utils import replace_layer_with_lora, count_lora_parameters, compute_em, compute_f1
from module.lora import LoRALinear, QuantLoRALinear


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='/data2/user/zhang/LLM/Falcon3-7B-Instruct-1.58bit', help='LLM Model Directory')
parser.add_argument('--epoch', type=int, default=5, help='Training Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size in Finetune')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning Rate')
parser.add_argument('--save_dir', type=str, default='./lora_ft_model/squad/Falcon3_158_7B_16r_vod_w6a8', help='Save Output LoRA Model Directory')
parser.add_argument('--q_proj', action='store_true', default=False, help='LoRA at Q Projection')
parser.add_argument('--k_proj', action='store_true', default=False, help='LoRA at K Projection')
parser.add_argument('--v_proj', action='store_true', default=False, help='LoRA at V Projection')
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

# Freeze Grad for All Parameters and Activate Grad Flow for LoRA Parameters
for param in model.parameters():
    param.requires_grad = False
for name, module in model.named_modules():
    if isinstance(module, (LoRALinear, QuantLoRALinear)):
        module.lora_a.requires_grad = True
        module.lora_b.requires_grad = True

# =====================
# Load SQuAD1.1 Dataset (Train and Validation)
# =====================
dataset = load_dataset("squad")

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


# Do not remove target and prompt in remove_columns
cols_to_remove = [col for col in dataset["train"].column_names if col not in ["target", "prompt"]]
tokenized_train = dataset["train"].map(preprocess_function, remove_columns=cols_to_remove)
tokenized_val = dataset["validation"].map(preprocess_function, remove_columns=cols_to_remove)

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
# Construct Data Loader
# =====================
train_dataloader = DataLoader(tokenized_train, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(tokenized_val, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

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
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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
        if global_step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                prompts_list = batch["prompts"][:2]
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
                    print(f"Step {global_step} sample {i + 1} - Prompt: {prompts_list[i]}")
                    print(f"Step {global_step} sample {i + 1} - Prediction: {pred}\n")
            model.train()
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss}")


    def clean_text(s):
        # Remove EOS Token
        return s.replace(tokenizer.eos_token, "").strip()

    model.eval()
    predictions = []
    targets = []
    prompts_list = []
    with torch.no_grad():
        for batch in val_dataloader:
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
            targets = batch["targets"]
            for gen in generated_ids:
                gen_text = tokenizer.decode(gen, skip_special_tokens=True)
                if "Answer: " in gen_text:
                    predicted_answer = gen_text.split("Answer:")[-1].strip()
                else:
                    predicted_answer = gen_text.strip()
                predictions.append(predicted_answer)
            # Probe 1 Batch of Data and Break
            break

    print("Validation predictions sample:")
    for i in range(min(3, len(predictions))):
        print(f"Sample {i + 1}:")
        print("Prompt:", prompts_list[i])
        print("Target:", targets[i])
        print("Prediction:", predictions[i])
        print("-" * 50)

    total_em = 0
    total_f1 = 0
    count = 0
    for tgt, pred in zip(targets, predictions):
        tgt_clean = clean_text(tgt)
        pred_clean = clean_text(pred)
        em = compute_em(pred_clean, tgt_clean)
        f1 = compute_f1(pred_clean, tgt_clean)
        total_em += em
        total_f1 += f1
        count += 1
    if count > 0:
        avg_em = total_em / count
        avg_f1 = total_f1 / count
        print(f"Validation EM: {avg_em:.4f}")
        print(f"Validation F1: {avg_f1:.4f}")
    else:
        print("No Samples For Evaluation.")

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
