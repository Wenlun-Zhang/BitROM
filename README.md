# BitROM
Verification on Adaptation Performance for "TODO"

## 1. Overview

* `./eval`: Script for evaluating adapter performance.
* `./module`: LoRA adapter modules.
* `./quantization`: Quantization functions.
* `./train`: Script for training LoRA adapters.
* `./utils`: Useful functions for this repository or analyzing BitNet models.

## 2. Environments

```
Python==3.10.16
numpy==2.2.3
torch==2.5.1+cu121
transformers==4.49.0
accelerate==1.4.0
lm-eval==0.3.0
```

## 3. BitROM

### 3.1. Abstract

TODO

### 3.2. To reproduce the main results in the paper

We use RTX A6000 GPU with 48G memory to run these adaptation experiments. To reproduce the main results, please download the Falcon3 BitNet base models at <a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026" target="_blank">Hugging Face</a>.

To fine-tune an adapter, run a script in `./train` by a corresponding task. You need to set arguments for a base model, training details, and adapter configurations. After fine-tuning an adapter, scripts in `./eval` are prepared for model evaluation. Please set identical adapter settings as fine-tuning, and specify the base or adaptation model with corresponding arguments.


