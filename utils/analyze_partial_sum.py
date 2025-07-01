import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.bitnet import BitLinear, unpack_weights
from collections import OrderedDict


layer_stats = OrderedDict()


def bitlinear_partial_hook(module, inputs, output):
    activations = inputs[0].detach()
    act_flat = activations.view(-1)
    act_min = float(act_flat.min().item())
    act_max = float(act_flat.max().item())

    packed_w = module.weight.detach().to(activations.device)
    unpacked = unpack_weights(packed_w, dtype=torch.int8)
    if isinstance(unpacked, (tuple, list)) and len(unpacked) >= 1:
        decoded_w = unpacked[0]
    else:
        decoded_w = unpacked
    decoded_w = decoded_w.to(activations.device).to(torch.int32)

    in_dim = decoded_w.size(1)
    flat_activations = activations.view(-1, in_dim).to(decoded_w.dtype)

    layer_name = module._get_name() + "@" + str(id(module))
    if layer_name not in layer_stats:
        layer_stats[layer_name] = [act_min, act_max, float("inf"), float("-inf")]
    else:
        prev_act_min, prev_act_max, prev_part_min, prev_part_max = layer_stats[layer_name]
        layer_stats[layer_name][0] = min(prev_act_min, act_min)
        layer_stats[layer_name][1] = max(prev_act_max, act_max)

    for x_vec in flat_activations:
        prods = decoded_w * x_vec.unsqueeze(0)
        partials = prods.cumsum(dim=1)

        curr_min = float(partials.min().item())
        curr_max = float(partials.max().item())

        prev_act_min, prev_act_max, prev_part_min, prev_part_max = layer_stats[layer_name]
        layer_stats[layer_name][2] = min(prev_part_min, curr_min)
        layer_stats[layer_name][3] = max(prev_part_max, curr_max)


def register_partial_hooks(model):
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.register_forward_hook(bitlinear_partial_hook)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe Max/Min Partial Sum of MAC Operation in BitLinear Layers")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained BitNet Model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device Selection")
    parser.add_argument("--use_tokenizer", action="store_true", help="Use Tokenizer to Transform Prompt_text into Input_ids, Otherwise Use Random Int Input")
    parser.add_argument("--prompt_text", type=str, default="Hello, world!", help="Tokenize Text if --use_tokenizer is active")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size in Inference")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence Length in Inference")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f">>> Load Model：{args.model_path} to {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}  # Confine to GPU0
    )
    model.to(device)
    model.eval()

    print(">>> Register Forward Hook for BitLinear Layers ...")
    register_partial_hooks(model)

    if args.use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_dir, padding_side="left")
        encoded = tokenizer(
            args.prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.seq_len
        )
        input_ids = encoded["input_ids"][:, : args.seq_len]
        input_ids = input_ids.repeat(args.batch_size, 1).to(device)
    else:
        vocab_size = model.config.vocab_size
        input_ids = torch.randint(
            low=0, high=vocab_size,
            size=(args.batch_size, args.seq_len),
            dtype=torch.long,
            device=device
        )

    print(f">>> Use Shape={input_ids.shape} Forward to Hook ...")
    with torch.no_grad():
        _ = model(input_ids)

    print("\n===== Activations Min/Max and Partial Sum Min/Max Values of BitLinear =====")
    print("Layer Name                                                   | act_min   act_max   | part_min   part_max")
    print("-" * 95)
    for layer_name, (act_min, act_max, part_min, part_max) in layer_stats.items():
        print(f"{layer_name:<60s} | {act_min:8.4f} {act_max:8.4f} | {part_min:10.4f} {part_max:10.4f}")
    print("===== Finish =====\n")


if __name__ == "__main__":
    main()
