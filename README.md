# BitROM
Verification on Adaptation Performance for "BitROM: Weight Reload-Free CiROM Architecture Towards Billion-Parameter 1.58-bit LLM Inference"

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

Compute-in-Read-Only-Memory (CiROM) accelerators offer outstanding energy efficiency for CNNs by eliminating runtime weight updates. However, their scalability to Large Language Models (LLMs) is fundamentally constrained by their vast parameter sizes. Notably, LLaMA-7B - the smallest model in LLaMA series - demands more than 1,000 cm2 of silicon area even in advanced CMOS nodes. This paper presents BitROM, the first CiROM-based accelerator that overcomes this limitation through co-design with BitNet's 1.58-bit quantization model, enabling practical and efficient LLM inference at the edge. BitROM introduces three key innovations: 1) a novel Bidirectional ROM Array that stores two ternary weights per transistor; 2) a Tri-Mode Local Accumulator optimized for ternary-weight computations; and 3) an integrated Decode-Refresh (DR) eDRAM that supports on-die KV-cache management, significantly reducing external memory access during decoding. In addition, BitROM integrates LoRA-based adapters to enable efficient transfer learning across various downstream tasks. Evaluated in 65nm CMOS, BitROM achieves 20.8 TOPS/W and a bit density of 4,967 kB/mm2 - offering a 10x improvement in area efficiency over prior digital CiROM designs. Moreover, the DR eDRAM contributes to a 43.6% reduction in external DRAM access, further enhancing deployment efficiency for LLMs in edge applications.

### 3.2. To reproduce the main results in the paper

We use RTX A6000 GPU with 48G memory to run these adaptation experiments. To reproduce the main results, please download the Falcon3 BitNet base models at <a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026" target="_blank">Hugging Face</a>.

To fine-tune an adapter, run a script in `./train` by a corresponding task. You need to set arguments for a base model, training details, and adapter configurations. After fine-tuning an adapter, scripts in `./eval` are prepared for model evaluation. Please set identical adapter settings as fine-tuning, and specify the base or adaptation model with corresponding arguments.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{ASiM,
  title={ASiM: Improving Transparency of SRAM-based Analog Compute-in-Memory Research with an Open-Source Simulation Framework},
  author={Zhang, Wenlun and Ando, Shimpei and Chen, Yung-Chin and Yoshioka, Kentaro},
  journal={arXiv:2411.11022},
  year={2024}
}
```
