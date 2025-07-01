import re, string, collections
import torch.nn
from transformers.integrations.bitnet import BitLinear
from module.lora import LoRALinear, QuantLoRALinear


def normalize_answer(s):
    """
    Normalize Context:
    Lower -> Remove Punctuation -> Remove Articles -> Remove Additional Space
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    """
    Compute F1 Scores between Predictions and Labels
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == "" and gt_norm == "":
        return 1.0
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_em(prediction, ground_truth):
    """
    Compute EM (Exact Match) Scores between Predictions and Labels
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def replace_layer_with_lora(model,
                            r=16,
                            alpha=32,
                            target_modules=None,
                            quant_lora=False,
                            x_bit=None,
                            lora_a_bit=None,
                            lora_b_bit=None):
    """
    Replace Original BitLinear Layer with LoRALinear/QuantLoRALinear
    target_modules: List of Strings. Replace All BitLinear Layers if None is Given
    quant_lora: LoRA Quantization Option
    """
    for name, module in model.named_modules():
        if isinstance(module, (BitLinear, torch.nn.Linear)):
            # Only layers in the module will be replaced if a list is provided
            if target_modules and not any(keyword in name for keyword in target_modules):
                continue
            sub_names = name.split('.')
            parent = model
            for sub_name in sub_names[:-1]:
                parent = getattr(parent, sub_name)
            orig_linear = getattr(parent, sub_names[-1])
            if quant_lora:
                assert None not in (x_bit, lora_a_bit, lora_b_bit), "Please Provide Quantization Bit Width."
                setattr(parent, sub_names[-1], QuantLoRALinear(orig_linear,
                                                               x_bit=x_bit,
                                                               lora_a_bit=lora_a_bit,
                                                               lora_b_bit=lora_b_bit,
                                                               r=r,
                                                               alpha=alpha))
            else:
                setattr(parent, sub_names[-1], LoRALinear(orig_linear,
                                                          r=r,
                                                          alpha=alpha))


def count_lora_parameters(model):
    """
    Calculate LoRA Parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, QuantLoRALinear)):
            lora_params += module.lora_a.numel() + module.lora_b.numel()
    ratio = lora_params / total_params * 100
    return lora_params, total_params, ratio


def load_lora_parameters_from_state_dict(model, state_dict, target_modules=None):
    """
    Load A/B Parameters to LoRALinear for Evaluation
    Load Parameters to Target Modules if target_modules is Given
    """
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, QuantLoRALinear)):
            if target_modules and not any(keyword in name for keyword in target_modules):
                continue
            key_a = name + ".lora_a"
            key_b = name + ".lora_b"
            if key_a in state_dict:
                module.lora_a.data.copy_(state_dict[key_a])
            if key_b in state_dict:
                module.lora_b.data.copy_(state_dict[key_b])
