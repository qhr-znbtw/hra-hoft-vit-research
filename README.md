# HRA and HOFT Research on ViT

A compact and reproducible benchmark for comparing Householder-based orthogonal adaptation with three standard transfer-learning baselines on a frozen ViT-B/16 backbone:

- **Linear Probe** — train only a task-specific classifier.
- **LoRA** — adapt the Query and Value projections with rank-8 low-rank updates.
- **Block OFT** — adapt the Query and Value projections with block-diagonal orthogonal transformations.
- **HRA** — right-adapt Query and Value with products of Householder reflections.
- **HOFT** — apply compact Householder transformations to both sides of Query and Value.

The repository uses the same backbone, data splits, optimizer family, schedule, preprocessing, and random seed across all methods. Its purpose is to provide a clean baseline for later research on orthogonal and parameter-efficient fine-tuning methods.

## Benchmark

The experiments use a three-task subset inspired by VTAB, chosen to cover different visual domains:

| Task | Domain | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| DTD | Natural textures | 47 | 800 | 200 | 1,880 |
| EuroSAT | Satellite imagery | 10 | 800 | 200 | 5,400 |
| dSprites Orientation | Structured synthetic images | 16 | 800 | 200 | 73,728 |

The pretrained backbone is `google/vit-base-patch16-224-in21k`. Adapter methods modify the Query and Value projections in all 12 Transformer encoder blocks while keeping the original pretrained parameters frozen.

## Results

All results below use one fixed seed (`42`) and the best validation checkpoint from a 100-epoch run.

| Method | DTD | EuroSAT | dSprites Orientation | Adapter parameters |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 0 |
| LoRA, rank 8, Q+V | 69.68% | 94.24% | **49.58%** | 294,912 |
| Block OFT, block size 24, Q+V | **70.27%** | 93.76% | 46.07% | 211,968 |
| HRA, rank 8, Q+V | 68.67% | 91.80% | 38.16% | **147,456** |
| HOFT, rank 4 per side, Q+V | 69.26% | **94.87%** | 46.86% | **147,456** |

HOFT gives the strongest EuroSAT result, LoRA is strongest on dSprites Orientation, and block OFT is strongest on DTD. HRA and HOFT use half as many adapter parameters as rank-8 LoRA. These are single-seed results and should be interpreted as a controlled implementation benchmark rather than a statistically conclusive ranking.

Detailed HRA/HOFT parameter counts, timings, peak GPU memory, and best epochs are available in the [research report](reports/hra_hoft_benchmark.md). The original three-method results remain documented in the [baseline report](reports/baseline_benchmark.md).

## Experimental Protocol

| Setting | Value |
| --- | --- |
| Backbone | ViT-B/16 pretrained on ImageNet-21k |
| Input size | 224 x 224 RGB |
| Epochs | 100 |
| Warmup | 10 epochs |
| Schedule | Linear warmup followed by cosine decay |
| Training batch size | 64 |
| Evaluation batch size | 128 |
| Optimizer | AdamW |
| Adapter learning rate | `1e-4` |
| Classification-head learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Loss | Cross-entropy |
| Precision | BF16 automatic mixed precision when supported |
| Data augmentation | None |
| Random seed | 42 |

## Repository Structure

```text
.
├── experiments/
│   ├── linear_probe/          # Frozen backbone and trainable classifier
│   ├── lora/                  # Rank-8 Query/Value LoRA
│   ├── oft/                   # Block-diagonal Query/Value OFT
│   ├── hra/                   # Rank-8 Query/Value HRA
│   └── hoft/                  # Rank-4 double-sided Query/Value HOFT
├── vit_core/                  # Shared ViT, data, training, and evaluation code
├── scripts/                   # Dataset and pretrained-weight preparation
├── tests/                     # Algebra, integration, and loading checks
├── results/                   # Formal 100-epoch result files
├── reports/                   # Benchmark report
├── data/                      # Local only; ignored by Git
├── pretrained/                # Local only; ignored by Git
└── checkpoints/               # Local only; ignored by Git
```

The ViT backbone and adaptation layers are implemented directly in PyTorch. Official Hugging Face weights are mapped into this transparent implementation rather than hiding the experiment behind a complete off-the-shelf classification model.

## Installation

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build from the [official installation guide](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data and Weights

Download DTD and EuroSAT:

```bash
python -m scripts.download_mini_vtab
```

Prepare dSprites Orientation:

```bash
python -m scripts.prepare_dsprites_orientation
```

Download the pretrained ViT-B/16 weights:

```bash
python -m scripts.download_pretrained_model
```

Datasets, pretrained weights, checkpoints, and model binaries are intentionally excluded from Git.

## Running the Experiments

Each entry point requires one of `dtd`, `eurosat`, or `dsprites_orientation`.

```bash
# Frozen-backbone linear probe
python -m experiments.linear_probe.run_linear_probe --task dtd

# LoRA
python -m experiments.lora.run_lora --task dtd

# Block OFT
python -m experiments.oft.run_oft --task dtd

# HRA
python -m experiments.hra.run_hra --task dtd

# HOFT
python -m experiments.hoft.run_hoft --task dtd
```

Replace `dtd` with another supported task to reproduce the remaining results. The best validation checkpoint and a JSON result summary are saved automatically.

## Verification

```bash
python -m tests.test_pretrained_loading
python -m tests.test_linear_probe
python -m tests.test_lora
python -m tests.test_oft
python -m tests.test_hra
python -m tests.test_hoft
```

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280)
- [Householder Reflection Adaptation](https://arxiv.org/abs/2405.17484)
- [HOFT: Householder Orthogonal Fine-tuning](https://arxiv.org/abs/2505.16531)
