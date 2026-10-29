# HRA and HOFT on Mini-VTAB-3

## Research question

This experiment asks whether Householder-based orthogonal adapters offer a
useful parameter/accuracy trade-off when adapting a frozen ViT-B/16. HRA and
HOFT are compared with three existing baselines under one controlled protocol.
This is an implementation-level comparison on our Mini-VTAB-3 suite, not an
exact reproduction of the datasets or hyperparameter searches in the original
papers.

## Methods

- **HRA:** each adapted projection is `W' = W H`, where `H` is a product of
  eight learned Householder reflections. An exact compact-WY representation is
  used in the forward pass.
- **HOFT:** each projection is adapted on both sides by Householder products.
  The forward pass uses the first-order compact-WY inverse from the official
  implementation. Rank 4 is used on each side.
- Both methods target Query and Value in all 12 Transformer blocks. The
  pretrained backbone remains frozen and the task classifier is trained.

For a 768 x 768 projection, HRA rank 8 and HOFT rank 4 both use 6,144 adapter
parameters. Across 24 Query/Value projections, each method therefore uses
147,456 adapter parameters. This matching is important: the comparison does
not give HOFT twice HRA's parameter budget merely because it acts on two sides.

## Protocol

| Setting | Value |
| --- | --- |
| Backbone | `google/vit-base-patch16-224-in21k` |
| Training/validation samples | 800 / 200 per task |
| Epochs | 100 |
| Warmup | 10 epochs |
| Schedule | Linear warmup, then cosine decay |
| Adapter/head learning rate | `1e-4` / `1e-3` |
| Optimizer | AdamW, weight decay `1e-4` |
| Batch size | 64 train, 128 evaluation |
| Precision | BF16 AMP on RTX 4070 Ti Super |
| Augmentation | None |
| Selection | Best validation checkpoint |
| Seed | 42 |

## Test accuracy

| Method | DTD | EuroSAT | dSprites Orientation | Mean |
| --- | ---: | ---: | ---: | ---: |
| Linear probe | 67.13% | 89.24% | 24.48% | 60.28% |
| LoRA, rank 8 | 69.68% | 94.24% | **49.58%** | **71.17%** |
| Block OFT, block size 24 | **70.27%** | 93.76% | 46.07% | 70.03% |
| HRA, rank 8 | 68.67% | 91.80% | 38.16% | 66.21% |
| HOFT, rank 4 per side | 69.26% | **94.87%** | 46.86% | 70.33% |

HOFT is the strongest method on EuroSAT and has the second-best mean among the
five methods. LoRA remains strongest on dSprites and on the three-task mean;
block OFT remains strongest on DTD. HRA improves on the linear probe but is
weaker than the other adapter methods in this single-seed experiment.

## Cost and checkpoint details

The trainable count below includes both adapter and task-head parameters, so it
changes slightly with the number of classes.

| Method | Task | Best epoch | Best val. | Trainable | Time | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRA | DTD | 72 | 69.00% | 183,599 | 374.23 s | 5.53 GiB |
| HRA | EuroSAT | 57 | 91.00% | 155,146 | 321.43 s | 5.52 GiB |
| HRA | dSprites Orientation | 84 | 37.00% | 159,760 | 317.61 s | 5.52 GiB |
| HOFT | DTD | 42 | 65.00% | 183,599 | 398.50 s | 5.98 GiB |
| HOFT | EuroSAT | 36 | 95.50% | 155,146 | 345.61 s | 5.98 GiB |
| HOFT | dSprites Orientation | 41 | 49.00% | 159,760 | 326.20 s | 5.98 GiB |

HOFT reaches its best checkpoint earlier and fits the 800-sample training sets
very quickly. On DTD and dSprites, its training accuracy reaches 100% while
validation accuracy plateaus or declines, indicating overfitting rather than an
optimization failure. HRA changes more conservatively but underfits the
dSprites task relative to LoRA and HOFT.

## Interpretation and limitations

The results do not support a universal ranking. Double-sided orthogonal
adaptation is particularly effective on EuroSAT, while the low-rank additive
update remains better for the structured dSprites task. The task-dependent
behavior is itself useful: preserving singular values is a constraint and an
inductive bias, not a guarantee of better transfer.

Only one seed and one shared hyperparameter setting were run. The 200-example
validation sets imply coarse accuracy increments and nontrivial selection
noise. A stronger study should add at least three seeds, report mean and
standard deviation, and test learning-rate or regularization changes suggested
by HOFT's visible overfitting.

## References

- [Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation](https://arxiv.org/abs/2405.17484)
- [HOFT: Householder Orthogonal Fine-tuning](https://arxiv.org/abs/2505.16531)
- [Official HOFT implementation](https://github.com/amorarc/HOFT)
