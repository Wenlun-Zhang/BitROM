import torch
import torch.nn as nn
from quantization.quant_utils import fake_uni_quantize_per_tensor, fake_ternary_weight_quantize_per_tensor


class LoRALinear(nn.Module):
    """
    Standard Floating-Point LoRA Module
    """
    def __init__(self,
                 layer,
                 r=16,
                 alpha=32):
        super().__init__()
        self.layer = layer
        in_features = layer.in_features
        out_features = layer.out_features

        self.lora_a = nn.Parameter(torch.randn(in_features, r, dtype=torch.bfloat16) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(r, out_features, dtype=torch.bfloat16))
        self.scale = alpha / r

    def forward(self, x):
        output = self.layer(x)
        lora_a_out = x @ self.lora_a.to(dtype=x.dtype, device=x.device)
        lora_out = lora_a_out @ self.lora_b.to(dtype=x.dtype, device=x.device) * self.scale
        return output + lora_out


class QuantLoRALinear(nn.Module):
    """
    LoRA Module with Quantization
    """
    def __init__(self,
                 layer,
                 x_bit=8,
                 lora_a_bit=8,
                 lora_b_bit=8,
                 r=16,
                 alpha=32):
        super().__init__()
        self.layer = layer
        in_features = layer.in_features
        out_features = layer.out_features

        self.lora_a = nn.Parameter(torch.randn(in_features, r, dtype=torch.bfloat16) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(r, out_features, dtype=torch.bfloat16))
        self.scale = alpha / r

        self.ternary_weight = False
        self.x_quant_min = -2 ** (x_bit - 1)
        self.x_quant_max = 2 ** (x_bit - 1) - 1
        if lora_a_bit == 1.58 and lora_b_bit == 1.58:
            self.ternary_weight = True
        else:
            self.lora_a_quant_min = -2 ** (lora_a_bit - 1)
            self.lora_a_quant_max = 2 ** (lora_a_bit - 1) - 1
            self.lora_b_quant_min = -2 ** (lora_b_bit - 1)
            self.lora_b_quant_max = 2 ** (lora_b_bit - 1) - 1
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

    def forward(self, x):
        x_scale = torch.abs(x).max() / (float(self.x_quant_max - self.x_quant_min) / 2)
        x_scale = torch.max(x_scale, self.eps)
        if self.ternary_weight:
            lora_a_scale = self.lora_a.abs().mean()
            lora_b_scale = self.lora_b.abs().mean()
        else:
            lora_a_scale = torch.abs(self.lora_a).max() / (float(self.lora_a_quant_max - self.lora_a_quant_min) / 2)
            lora_b_scale = torch.abs(self.lora_b).max() / (float(self.lora_b_quant_max - self.lora_b_quant_min) / 2)
        lora_a_scale = torch.max(lora_a_scale, self.eps)
        lora_b_scale = torch.max(lora_b_scale, self.eps)

        output = self.layer(x)

        x = fake_uni_quantize_per_tensor(x, x_scale, self.x_quant_min, self.x_quant_max)

        if self.ternary_weight:
            lora_a = fake_ternary_weight_quantize_per_tensor(self.lora_a, lora_a_scale)
            lora_b = fake_ternary_weight_quantize_per_tensor(self.lora_b, lora_b_scale)
        else:
            lora_a = fake_uni_quantize_per_tensor(self.lora_a, lora_a_scale, self.lora_a_quant_min, self.lora_a_quant_max)
            lora_b = fake_uni_quantize_per_tensor(self.lora_b, lora_b_scale, self.lora_b_quant_min, self.lora_b_quant_max)

        lora_a_out = x @ lora_a.to(dtype=x.dtype, device=x.device)
        lora_out = lora_a_out @ lora_b.to(dtype=x.dtype, device=x.device) * self.scale

        return output + lora_out
