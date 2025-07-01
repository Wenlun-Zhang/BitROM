import argparse
import torch
from collections import OrderedDict
from transformers import AutoModelForCausalLM
from transformers.integrations.bitnet import BitLinear, unpack_weights


def longest_run_of_one_or_minus_one_1d(arr: torch.Tensor) -> int:
    flat = arr.view(-1).cpu().numpy()
    max_run = 0
    current_run = 0
    current_sign = 0

    for xi in flat:
        if xi == 1 or xi == -1:
            if xi == current_sign:
                current_run += 1
            else:
                current_sign = int(xi)
                current_run = 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_sign = 0
            current_run = 0

    return int(max_run)


def collect_bitlinear_runs(model: torch.nn.Module, device: torch.device) -> OrderedDict:

    runs = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            packed_w = module.weight.to(device)

            unpacked = unpack_weights(packed_w, dtype=torch.int8)
            if isinstance(unpacked, (tuple, list)) and len(unpacked) >= 1:
                decoded_w = unpacked[0]
            else:
                decoded_w = unpacked

            decoded_w = decoded_w.to(device)

            max_run = longest_run_of_one_or_minus_one_1d(decoded_w)
            runs[name] = max_run

    return runs


def main():
    parser = argparse.ArgumentParser(description="Analyze Longest Sequence of +-1 Weight")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained BitNet Model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device Selection")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading '{args.model_path}' to {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}  # Confine to GPU0
    )
    model.to(device)
    model.eval()

    print("Traverse BitLinear Layers，Analyze Continuous +1/-1 Sequence Length ...")
    bitlinear_runs = collect_bitlinear_runs(model, device)

    print("\n===== BitLinear +1/-1 Sequence Length =====")
    for layer_name, max_run in bitlinear_runs.items():
        print(f"{layer_name:<60s} → Longest +1/-1 Subsequence Length = {max_run}")
    print("===== Finish =====")


if __name__ == "__main__":
    main()
