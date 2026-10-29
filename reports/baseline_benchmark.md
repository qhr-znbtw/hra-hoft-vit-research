# ViT Transfer-Learning Baseline Report

## 1. Objective

This benchmark establishes three controlled reference methods for later parameter-efficient fine-tuning research:

1. Linear Probe: freeze the ViT representation and train only the classifier.
2. LoRA: learn low-rank updates to attention projections.
3. Block OFT: learn orthogonal transformations of attention projections.

The comparison is designed to isolate the adaptation method. All three experiments share the same pretrained ViT-B/16, task splits, preprocessing, training duration, validation procedure, optimizer family, schedule, and random seed.

## 2. Tasks and Protocol

The benchmark covers three visual domains:

| Task | Domain | Classes | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| DTD | Natural textures | 47 | 800 | 200 | 1,880 |
| EuroSAT | Specialized satellite imagery | 10 | 800 | 200 | 5,400 |
| dSprites Orientation | Structured synthetic images | 16 | 800 | 200 | 73,728 |

Common settings:

- backbone: `google/vit-base-patch16-224-in21k`;
- resolution: 224 x 224 RGB;
- preprocessing: resize, tensor conversion, and normalization to mean/std 0.5;
- epochs: 100;
- warmup: 10 epochs followed by cosine learning-rate decay;
- batch sizes: 64 for training and 128 for evaluation;
- optimizer: AdamW;
- weight decay: `1e-4`;
- classifier learning rate: `1e-3`;
- adapter learning rate: `1e-4`;
- loss: cross-entropy;
- seed: 42;
- no task-specific data augmentation;
- checkpoint selection: highest validation accuracy.

LoRA and OFT are applied to the Query and Value projections in every Transformer encoder block. The original pretrained parameters remain frozen.

## 3. Methods

### 3.1 Linear Probe

Only the task-specific linear classifier is optimized. It measures how linearly separable each target task already is in the pretrained representation.

### 3.2 LoRA

For a frozen projection matrix \(W\), LoRA learns a low-rank update

\[
W' = W + \frac{\alpha}{r}BA,
\]

with rank \(r=8\) and \(\alpha=8\). Query and Value are adapted in all 12 encoder blocks.

### 3.3 Block OFT

For a frozen projection matrix \(W\), OFT learns an orthogonal transformation \(R\) and uses

\[
W' = WR^T.
\]

The 768-dimensional feature space is divided into 32 blocks of size 24. Each block is parameterized through a skew-symmetric matrix and the Cayley transform, so the learned transformation remains orthogonal.

## 4. Accuracy

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear Probe | 67.13% | 89.24% | 24.48% | 60.28% |
| LoRA | 69.68% | **94.24%** | **49.58%** | **71.17%** |
| Block OFT | **70.27%** | 93.76% | 46.07% | 70.03% |

Both adapter methods substantially improve the structured dSprites task, showing that a frozen linear classifier cannot fully recover the target representation. LoRA has the highest mean accuracy, while Block OFT is strongest on DTD.

## 5. Parameter Efficiency

### Adapter parameters

| Method | Adapter parameters |
| --- | ---: |
| Linear Probe | 0 |
| LoRA | 294,912 |
| Block OFT | 211,968 |

Block OFT uses 82,944 fewer adapter parameters than LoRA, a reduction of 28.1%.

### Total trainable parameters

The classifier size changes with the number of task classes.

| Method | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Linear Probe | 36,143 | 7,690 | 12,304 |
| LoRA | 331,055 | 302,602 | 307,216 |
| Block OFT | 248,111 | 219,658 | 224,272 |

## 6. Efficiency Measurements

All measurements were collected on one NVIDIA GeForce RTX 4070 Ti SUPER.

### Training time

| Method | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Linear Probe | 224.8 s | 174.7 s | 183.0 s |
| LoRA | 318.1 s | 286.7 s | 284.2 s |
| Block OFT | 370.1 s | 324.7 s | 320.8 s |

### Peak GPU memory

| Method | Peak memory |
| --- | ---: |
| Linear Probe | 1.29 GiB |
| LoRA | 5.52 GiB |
| Block OFT | 5.52 GiB |

### Best validation epoch

| Method | DTD | EuroSAT | dSprites Orientation |
| --- | ---: | ---: | ---: |
| Linear Probe | 39 | 69 | 75 |
| LoRA | 34 | 61 | 61 |
| Block OFT | 44 | 69 | 54 |

## 7. Interpretation

The results support three practical conclusions:

1. Linear probing is inexpensive and competitive on EuroSAT, but it is insufficient for dSprites Orientation.
2. LoRA provides the strongest average accuracy in this protocol and is a useful general PEFT baseline.
3. Block OFT reaches comparable performance with fewer adapter parameters, at the cost of somewhat longer training caused by constructing orthogonal factors.

The benchmark does not establish universal superiority. It contains only three tasks and one random seed, and the methods are not exactly parameter matched. Multiple seeds, confidence intervals, and parameter-matched configurations are appropriate extensions.

## 8. Reproduction

After preparing the datasets and pretrained weights, run:

```bash
python -m experiments.linear_probe.run_linear_probe --task dtd
python -m experiments.lora.run_lora --task dtd
python -m experiments.oft.run_oft --task dtd
```

Repeat each command with `eurosat` and `dsprites_orientation`. Machine-readable outputs for the completed 100-epoch runs are stored under `results/<method>/development_800_200/<task>/seed_42.json`.
