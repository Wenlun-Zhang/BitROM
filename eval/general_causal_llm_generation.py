import sys
sys.path.append("/home/zhang/Project/BitROM")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import replace_layer_with_lora, count_lora_parameters, load_lora_parameters_from_state_dict
from safetensors.torch import load_file as load_safetensors

device = "cuda" if torch.cuda.is_available() else "cpu"

eval_lora = False

# Model Directory
base_dir = "/data2/user/zhang/LLM/Falcon3-1B-Instruct-1.58bit"
lora_dir = "/home/zhang/Project/AdaROM/output/lora_ft_model/squad20/7B_16r_vod_6b"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, device_map={"": 0})
print("Model & Tokenizer loaded successfully!")

if eval_lora:
    target_modules = ["v_proj", "o_proj", "down_proj"]
    replace_layer_with_lora(model,
                            r=16,
                            alpha=32,
                            target_modules=target_modules,
                            quant_lora=True,
                            x_bit=8,
                            lora_a_bit=6,
                            lora_b_bit=6)
    lora_params, total_params, lora_ratio = count_lora_parameters(model)
    print(f"LoRA parameters: {lora_params}, Total parameters: {total_params}, Ratio: {lora_ratio:.2f}%")
    # Load LoRA Parameters from checkpoint
    state_dict = load_safetensors(os.path.join(lora_dir, "model.safetensors"))
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    load_lora_parameters_from_state_dict(model, state_dict, target_modules=target_modules)


def generate_response(prompt, max_length=2048, num_beams=5, temperature=0.7):

    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        early_stopping=True,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Generation Test
prompt = """Question: Introduce the history of Micron Technology, Inc.
Answer:"""

response = generate_response(prompt)

print("Model Response:\n", response)
