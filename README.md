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

Compute-in-Read-Only-Memory (CiROM) accelerators offer outstanding energy efficiency for CNNs by eliminating runtime weight updates, but they are fundamentally limited in scaling to Large Language Models (LLMs) due to excessive silicon area demands. Even relatively compact models such as LLaMA-7B require more than 1000 cm2 in advanced CMOS nodes. This paper presents BitROM, the first CiROM-based accelerator that overcomes this limitation through co-design with BitNet’s 1.58-bit quantization model, enabling practical and efficient LLM inference at the edge. BitROM introduces three key innovations: 1) a novel Bidirectional ROM Array that stores two ternary weights per transistor; 2) a Tri-Mode Local Accumulator optimized for ternary-weight computations; and 3) an integrated Decode-Refresh eDRAM that supports on-die KV cache management, significantly reducing external memory access during decoding. In addition, BitROM integrates LoRA-based adapters to enable efficient transfer learning across various downstream tasks. Evaluated in 65nm CMOS, BitROM achieves 20.8 TOPS/W and a bit density of 4967 kB/mm2—offering a 10$\times$ improvement in area efficiency over prior digital CiROM designs. Moreover, the on-chip KV-cache contributes to a 43.6% reduction in DRAM access, further enhancing deployment efficiency for LLMs at the edge.

### 3.2. To reproduce the main results in the paper

We use RTX A6000 GPU with 48G memory to run these adaptation experiments. To reproduce the main results, please download the Falcon3 BitNet base models at <a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026" target="_blank">Hugging Face</a>.

To fine-tune an adapter, run a script in `./train` by a corresponding task. You need to set arguments for a base model, training details, and adapter configurations. After fine-tuning an adapter, scripts in `./eval` are prepared for model evaluation. Please set identical adapter settings as fine-tuning, and specify the base or adaptation model with corresponding arguments.


