import torch


def round_ste(x: torch.Tensor):
    """
    STE for Quantization Training
    """
    return (x.round() - x).detach() + x


def fake_uni_quantize_per_tensor(x, scale, quant_min, quant_max):
    """
    Per-Tensor Uniform Quantization
    """
    xq = torch.clamp(round_ste(x / scale), quant_min, quant_max) * scale
    return xq


def fake_log_quantize_per_tensor(x, scale, quant_min, quant_max):
    """
    Per-Tensor Log2 Quantization
    """
    levels = quant_max - quant_min + 1
    x = torch.clamp(x, 1e-20, None)
    xq = round_ste(-1 * (x / scale).log2())
    softmax_mask = (xq >= levels)
    xq = torch.clamp(xq, 0, levels - 1)
    xq = scale * 2 ** (-1 * xq)
    xq[softmax_mask] = torch.Tensor([0.0])
    return xq


def fake_ternary_weight_quantize_per_tensor(weight, scale):
    """
    Per-Tensor Ternary Weight Quantization
    """
    dtype = weight.dtype
    xq = torch.clamp(round_ste(weight / scale), -1, 1) * scale
    return xq.type(dtype)
