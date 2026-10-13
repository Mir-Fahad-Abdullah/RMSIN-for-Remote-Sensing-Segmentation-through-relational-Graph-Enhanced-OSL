# Referring Remote Sensing Image Segmentation through a Relational Graph-Enhanced Open-Set Network


## 📖 Overview

Referring remote sensing image segmentation takes an aerial or satellite image plus a sentence such as *"the gray and black ship on the right"* and returns a pixel-level mask of exactly that object.

Existing methods share a **closed-set assumption**: every object class seen at test time was also seen during training. Real remote sensing pipelines don't work that way, because analysts keep running into new object types.

This project tackles both problems at once:

1. **Better cross-modal fusion.** We replace RMSIN's Pixel-Word Attention Module (PWAM) with a custom **Heterogeneous Graph Neural Network (HeteroGNN)** that treats image and text tokens as nodes of one graph.
2. **Open-set rejection.** We add **Adversarial Reciprocal Point Learning (ARPL)** and five out-of-distribution scoring functions so the model can reject unknown classes at inference time.
3. **Explainability.** Three tools show where the model looks, which words matter, and where it is uncertain.

## ✨ Highlights

- **74.94% oIoU / 61.32% mIoU** on RRSIS-D (ConvNeXt-V2 + HeteroGNN), about **8 points above** the Swin-V1 baseline under the same training setup.
- **HeteroGNN fusion**: two rounds of six message-passing operations over four edge types (L→L, L→V, V→V, V→L) with a learned residual gate.
- **Backbone study** across Swin-V1, Swin-V2, ConvNeXt-V2, FocalNet, and PVT-V2.
- **Open-set evaluation** on three semantic-aware known/unknown class splits.
- **Explainability suite**: SegGradCAM, cross-modal attention maps, and rejection (uncertainty) heatmaps.

## 🏗️ Architecture

The pipeline is built on [RMSIN](https://arxiv.org/abs/2306.08625) (rotated multi-scale interaction network). Everything downstream of the fusion block is inherited unchanged.

```
Image ──► ConvNeXt-V2 (stages 0–3) ──┐
                                      ├─► HeteroGNN fusion (at every stage) ─► Cross-scale Interaction Module
Text ───► BERT-base-uncased ─────────┘                                                    │
                                                                                          ▼
                         ARPL open-set head ◄── pooled features        SimpleDecoding + Adaptive Rotated Conv ─► Mask
                         (known / reject)                                                  │
                                                                                          ▼
                                                                                  SegGradCAM / XAI tools
```
<p align="center">
<img width="996" height="514" alt="image" src="https://github.com/user-attachments/assets/e0425f62-e89a-4e7e-9f66-cc3ac4707016" />
</p>

### HeteroGNN fusion block

Visual tokens `h_v` and language tokens `h_l` are projected into a shared space, then refined by six operations per round:

| Step | Operation | Stream |
|:-:|---|---|
| 1 | Language self-attention | language |
| 2 | Language feed-forward (L-FFN) | language |
| 3 | Language → vision cross-attention | vision attends to language |
| 4 | Vision self-attention | vision |
| 5 | Vision → language cross-attention | language attends to vision |
| 6 | Vision feed-forward (V-FFN) | vision |

After two rounds, a learned gate blends the fused and original visual features with a residual skip connection, so the model falls back to the visual features when fusion doesn't help:

```
o = g ⊙ h_v + (1 − g) ⊙ v,    g = σ(MLP(h_v))
```

### Open-set learning (ARPL)

- Visual features are globally pooled into a 1024-d vector per sample and fed to an ARPL module with **20 reciprocal points** and 20 learnable radii.
- Training objective: `L = L_seg + 0.1 · L_ARPL`, where `L_seg` mixes cross-entropy (class weights [0.9, 1.1]) and Dice loss (w = 0.1), and `L_ARPL` combines classification, adversarial margin, and inter-class separation losses.
- **Five OOD scorers** are computed at inference: MSP, MaxLogit, Energy, Prototype Distance, and the ARPL score.
- **Metrics**: AUROC, FPR@95, harmonic mean H of mIoU and (1 − FPR@95), and OSCR AUC.

<img width="526" height="443" alt="image" src="https://github.com/user-attachments/assets/2309c6ab-4ec7-41ad-bd4b-44f4fb9e0ff0" />



## 📦 Dataset

**RRSIS-D** (from the RMSIN authors): 17,402 image–caption–mask triplets, 800×800 px, 0.5–30 m ground resolution, 20 object categories, annotated semi-automatically with SAM guidance.

| Split | Samples |
|---|:-:|
| Train | 10,441 |
| Validation | 3,480 |
| Test | 3,481 |

**Semantic-aware open-set splits.** The 20 categories are grouped into five semantic groups by function and visual similarity, so unknown classes are not near-duplicates of known ones. Each split holds out 5 unknown categories (15 known, 5 unknown), such as:

| Split | Unknown categories |
|:-:|---|
| 0 | vehicle, chimney, storage tank, stadium, harbor |
| 1 | vehicle, windmill, bridge, stadium, harbor |
| 2 | vehicle, chimney, dam, stadium, harbor |

## 📊 Results

### Fusion module and backbone (RRSIS-D test set)

| Backbone | Fusion | oIoU (%) | mIoU (%) | Pr@0.5 | Pr@0.7 | Pr@0.9 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Swin-V1 | PWAM (RMSIN baseline, our reproduction) | 66.65 | 50.21 | 57.64 | 38.56 | 10.87 |
| Swin-V1 | HeteroGNN | 67.05 | 51.09 | 58.33 | 39.09 | 11.04 |
| ConvNeXt-V2 | PWAM | 74.61 | 60.63 | 70.55 | 50.59 | 18.85 |
| **ConvNeXt-V2** | **HeteroGNN** | **74.94** | **61.32** | **70.90** | **52.86** | **21.34** |
| PVT-V2 | HeteroGNN | 74.78 | 60.58 | 70.38 | 52.69 | 22.38 |

- **ConvNeXt-V2 gives the biggest jump** (about +8 oIoU over Swin-V1), regardless of fusion module.
- **HeteroGNN gives consistent gains** at every precision threshold, with the clearest benefit at high IoU.
- **PVT-V2 is a strong lightweight alternative**: nearly identical accuracy with 193M parameters versus 329M.
- **HeteroGNN beats a GraphSAGE-style variant** by about 3.2 oIoU and 3.5 mIoU.

### HeteroGNN ablation (ConvNeXt-V2)

| Configuration | oIoU (%) | mIoU (%) |
|---|:-:|:-:|
| No V→V self-attention | 73.30 | 58.96 |
| Strided-pooling V→V | 73.27 | 60.19 |
| 1 round | 72.77 | 57.46 |
| **2 rounds (selected)** | **74.94** | **61.32** |
| 3 rounds | 73.29 | 59.92 |

### Open-set detection (mean ± std over 3 splits)

| Scorer | AUROC ↑ | FPR@95 ↓ | Harmonic mean ↑ |
|---|:-:|:-:|:-:|
| MSP | 0.283 ± 0.066 | 0.996 | 0.008 |
| MaxLogit / Energy | 0.295 ± 0.070 | 0.965 | 0.065 |
| ARPL score | 0.325 ± 0.045 | 0.998 | 0.004 |
| **Prototype Distance** | **0.703 ± 0.035** | **0.781** | **0.325** |

**Key finding.** The logit-based scorers fall *below chance*. Because ARPL's logits are built from distances to reciprocal points, unknown samples (far from everything) end up looking *more* confident than known ones, a **distance-inversion effect**. Scoring in the 1024-d feature space with **prototype distance** avoids this and works consistently across all three splits (AUROC 0.66–0.74). Joint ARPL training does not hurt closed-set segmentation (73.42% test oIoU).

## 🔍 Explainability

| Tool | What it shows |
|---|---|
| **SegGradCAM** | Spatial heatmap of the image regions that drove the foreground mask |
| **Cross-modal attention maps** | Which words in the expression attend to which image regions |
| **Rejection GradCAM** | Uncertainty heatmaps from the MSP, MaxLogit, and Energy scorers |

## ⚙️ Training Setup

| | |
|---|---|
| Hardware | NVIDIA A100 40 GB (Google Colab Pro), BF16 mixed precision |
| Optimizer | AdamW, weight decay 0.01, LR 3e-5, polynomial decay (power 0.9) |
| Batch size | 16–24 |
| Epochs | up to 40, early stopping (patience 5) on validation oIoU |
| ConvNeXt-V2 extras | layer-wise LR decay 0.7, stochastic depth 0.1, 2 frozen stages |
| Text encoder | BERT-base-uncased (layers 10–11 and embeddings frozen) |
| Input size | 480 × 480 |


## 👥 Authors

| Name | Affiliation |
|---|---|
| Fahim Shahriar | Dept. of ECE, North South University |
| Mir Fahad Abdullah | Dept. of ECE, North South University |
| Jannatul Ferdous | Dept. of ECE, North South University |
