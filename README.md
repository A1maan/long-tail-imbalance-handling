# Long-Tail Class Imbalance Handling for Object Detection

A research-oriented Python toolkit for studying and mitigating **long-tail class imbalance** in object detection models, evaluated on the **COCO val2017** benchmark. This repository implements and compares multiple strategies — from vanilla baselines to custom loss functions, model ensembles, and class-specific confidence thresholding — to improve detection performance on rare (tail) classes without sacrificing head-class accuracy.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Approaches](#approaches)
  - [1. Baseline](#1-baseline)
  - [2. Custom Loss Functions](#2-custom-loss-functions)
  - [3. Model Ensemble (WBF)](#3-model-ensemble-wbf)
  - [4. Class-Specific Confidence Thresholding](#4-class-specific-confidence-thresholding)
- [Results](#results)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Setup](#dataset-setup)
- [Key Concepts](#key-concepts)
- [References](#references)

---

## Overview

Real-world object detection datasets like COCO exhibit a **long-tail distribution**: a small number of "head" classes (e.g., *person*, *car*) dominate the training data, while many "tail" classes (e.g., *toothbrush*, *hair drier*, *scissors*) are severely under-represented. Standard detectors trained with uniform loss functions are biased toward head classes, yielding poor performance on tail classes at test time.

This project investigates four complementary mitigation strategies and benchmarks them using the official COCO evaluation protocol (mAP@[0.5:0.95], mAP@0.5, mAP@0.75), broken down into **head**, **medium**, and **tail** class groups.

---

## Problem Statement

| Class Group | Example Classes | Characteristics |
|---|---|---|
| **Head** | person, car, dog, cat, bicycle, chair | High frequency, well-represented in training |
| **Medium** | airplane, bus, train, bottle, cup, etc. | Moderate frequency |
| **Tail** | toothbrush, hair drier, scissors, keyboard, mouse, oven, toaster | Rare, under-represented — hardest to detect |

Standard BCE-based detectors over-predict head classes and under-detect tail classes. This repo tackles this via loss reweighting, ensembling, and inference-time thresholding.

---

## Repository Structure

```
long-tail-imbalance-handling/
│
├── baseline/                          # Vanilla multi-model inference & evaluation
│   ├── baseline_inference.py          # Run inference across all supported models
│   ├── calculate_class_metrics.py     # Per-class & head/medium/tail AP computation
│   ├── visualize_predictions.py       # Visualize model predictions on COCO images
│   ├── yolo_to_coco_mapping.json      # YOLO class index → COCO category ID mapping
│   ├── requirements.txt               # Dependencies for this module
│   └── runs/                          # Output directory for inference results
│
├── loss-functions/                    # Custom loss function training
│   ├── FocalLoss.py                   # Focal Loss implementation (Lin et al., 2017)
│   ├── VarifocalLoss.py               # Varifocal Loss implementation (Zhang et al., 2020)
│   ├── custom_trainer.py              # Custom Ultralytics trainer with Focal/VFL loss
│   ├── train.py                       # Training script for YOLO + RT-DETR models
│   ├── train_torchvision.py           # Training for TorchVision models (Faster R-CNN, RetinaNet)
│   └── yolo_to_coco_mapping.json      # YOLO → COCO class mapping
│
├── ensemble/                          # Multi-model ensemble via Weighted Boxes Fusion
│   ├── ensemble_inference_val2017.py  # Run WBF ensemble inference on COCO val2017
│   ├── calculate_ensemble_metrics.py  # Compute ensemble AP metrics (head/medium/tail)
│   ├── compare_baseline_ensemble.py   # Compare baseline vs ensemble with visualizations
│   ├── filter_and_benchmark_ensemble.py # Sweep confidence thresholds for ensemble
│   ├── yolo_to_coco_mapping.json      # YOLO → COCO class mapping
│   ├── requirements.txt               # Dependencies for this module
│   ├── predictions/                   # Stored ensemble predictions & metrics JSONs
│   │   ├── ensemble_metrics.json
│   │   ├── ensemble_summary.json
│   │   └── ensemble_class_metrics.json
│   └── comparisons/                   # HTML/chart outputs from comparison scripts
│       └── metrics_comparison.html
│
├── class-specific-conf-threshold/     # Per-class inference threshold tuning
│   └── yolo_to_coco_mapping.json      # YOLO → COCO class mapping
│
├── .gitignore
└── README.md
```

---

## Approaches

### 1. Baseline

The **baseline** module runs off-the-shelf pre-trained models on the COCO val2017 set and evaluates performance with the standard COCO API, additionally breaking down AP by long-tail group (head / medium / tail).

**Key files:**
- `baseline_inference.py` — Orchestrates multi-model inference, handles YOLO-to-COCO label remapping, and stores raw predictions
- `calculate_class_metrics.py` — Aggregates per-class AP into head/medium/tail summaries using `pycocotools`
- `visualize_predictions.py` — Draws bounding boxes, confidence scores, and class labels onto COCO images for qualitative inspection

**Models benchmarked:**
- YOLOv8s, YOLO11s, YOLO12s
- RT-DETR-L
- Faster R-CNN (ResNet-50-FPN v2)
- RetinaNet (ResNet-50-FPN v2)

---

### 2. Custom Loss Functions

The **loss-functions** module replaces the default binary cross-entropy (BCE) classification loss in YOLO and RT-DETR with two class-imbalance-aware alternatives.

#### Focal Loss

> Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017. [[arXiv]](https://arxiv.org/abs/1708.02002)

Focal Loss down-weights the loss contribution of easy, well-classified examples and focuses training on hard negatives:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **γ (gamma):** Focusing parameter — higher values increase focus on hard examples (default: `2.0` for YOLO, `1.5` for RT-DETR)
- **α (alpha):** Class balancing factor (default: `0.25`)

**Integration:** `FLDetectionModel` (YOLO) and `FLRTDETRModel` (RT-DETR) subclass the corresponding Ultralytics model classes and swap `criterion.bce` for `FocalLoss` during `init_criterion()`.

#### Varifocal Loss

> Zhang et al., *"VarifocalNet: An IoU-aware Dense Object Detector"*, CVPR 2021. [[arXiv]](https://arxiv.org/abs/2008.13367)

Varifocal Loss extends Focal Loss with **asymmetric weighting** — positive samples are weighted by their IoU-aware classification score (`gt_score`), while negatives are down-weighted by `p^\gamma`:

$$VFL(p, q) = \begin{cases} -q(q \log(p) + (1-q)\log(1-p)) & q > 0 \\\ -\alpha p^\gamma \log(1-p) & q = 0 \end{cases}$$

This encourages the model to produce high-quality, IoU-calibrated confidence scores — particularly useful for tail classes that rarely achieve high IoU.

**TorchVision support:** `train_torchvision.py` adds Focal Loss to Faster R-CNN (by patching the classification head loss) and trains RetinaNet (which natively uses Focal Loss) with DDP multi-GPU support.

---

### 3. Model Ensemble (WBF)

The **ensemble** module fuses predictions from multiple heterogeneous detectors using **Weighted Boxes Fusion (WBF)**.

> Solovyev et al., *"Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models"*, 2020. [[arXiv]](https://arxiv.org/abs/1910.13302)

Unlike NMS-based merging, WBF aggregates all candidate boxes proportionally to their confidence scores, preserving more true positives for rare classes.

**Detector pool:**
- YOLOv8s, YOLO11s, YOLO12s
- RT-DETR-L
- Faster R-CNN v2
- RetinaNet

**Key files:**
- `ensemble_inference_val2017.py` — Loads all detectors, runs parallel inference on COCO val2017, and applies WBF per image
- `calculate_ensemble_metrics.py` — Evaluates ensemble predictions using `pycocotools.COCOeval`
- `filter_and_benchmark_ensemble.py` — Sweeps confidence thresholds (e.g., 0.2–0.7) to find the optimal post-ensemble filter
- `compare_baseline_ensemble.py` — Generates side-by-side bar charts and HTML comparison tables

**Ensemble results (WBF, COCO val2017):**

| Metric | Value |
|---|---|
| mAP@[0.5:0.95] | 0.432 |
| mAP@0.5 | 0.560 |
| mAP@0.75 | 0.478 |
| Head AP | 0.460 |
| Medium AP | 0.434 |
| Tail AP | 0.438 |

> The WBF ensemble improves tail-class AP and provides more balanced head/medium/tail performance compared to any single model.

---

### 4. Class-Specific Confidence Thresholding

The **class-specific-conf-threshold** module applies **per-class confidence thresholds** during inference, rather than a single global threshold. Tail classes are assigned lower thresholds to improve their recall, while head classes retain higher thresholds to keep precision in check.

This is an inference-time, training-free technique that complements any of the above approaches.

---

## Results

Baseline model performance on COCO val2017 (mAP@[0.5:0.95]):

| Model | Overall mAP | mAP@0.5 | mAP@0.75 |
|---|---|---|---|
| RT-DETR-L | **0.506** | 0.679 | 0.549 |
| Faster R-CNN | 0.441 | 0.626 | 0.488 |
| YOLO12s | 0.409 | 0.533 | 0.447 |
| YOLO11s | 0.406 | 0.533 | 0.443 |
| YOLOv8s | 0.396 | 0.525 | — |
| RetinaNet | 0.333 | 0.460 | 0.365 |
| **Ensemble (WBF)** | **0.432** | **0.560** | **0.478** |

> The WBF ensemble improves tail-class AP and provides more balanced head/medium/tail performance compared to any single model.

---

## Supported Models

| Model | Framework | Notes |
|---|---|---|
| YOLOv8 (n/s/m/l/x) | Ultralytics | All sizes supported |
| YOLO11 (all sizes) | Ultralytics | — |
| YOLO12 (all sizes) | Ultralytics | — |
| RT-DETR (l, x) | Ultralytics | Transformer-based |
| Faster R-CNN ResNet-50-FPN v2 | TorchVision | — |
| RetinaNet ResNet-50-FPN v2 | TorchVision | Native Focal Loss |

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (recommended)
- COCO val2017 dataset

### Install dependencies

Each module has its own `requirements.txt`. Install for the module(s) you intend to use:

```bash
# Baseline
pip install -r baseline/requirements.txt

# Ensemble
pip install -r ensemble/requirements.txt
```

**Core dependencies:**

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
pycocotools>=2.0.6
ensemble-boxes>=1.0.8
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=9.0.0
matplotlib
seaborn
tqdm
scikit-learn
pandas
scipy
albumentations
tensorboard
```

---

## Usage

### Baseline Inference

```bash
cd baseline
python baseline_inference.py
```

### Calculate Per-Class Metrics

```bash
cd baseline
python calculate_class_metrics.py
```

### Visualize Predictions

```bash
cd baseline
python visualize_predictions.py
```

### Train with Focal Loss (YOLO / RT-DETR)

```bash
cd loss-functions
python train.py
```

### Train with Focal Loss (TorchVision — Faster R-CNN / RetinaNet)

```bash
cd loss-functions
# Single GPU
python train_torchvision.py

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 train_torchvision.py
```

### Run Ensemble Inference

```bash
cd ensemble
python ensemble_inference_val2017.py
```

### Evaluate Ensemble

```bash
cd ensemble
python calculate_ensemble_metrics.py
```

### Sweep Confidence Thresholds

```bash
cd ensemble
python filter_and_benchmark_ensemble.py --thresholds 0.2 0.3 0.4 0.5 0.6 0.7
```

### Compare Baseline vs Ensemble

```bash
cd ensemble
python compare_baseline_ensemble.py \
  --baseline /path/to/baseline/metrics_summary.json \
  --ensemble predictions/ensemble_summary.json \
  --output comparisons/
```

---

## Dataset Setup

This project evaluates on **COCO val2017**.

1. Download the COCO val2017 images and annotations from [https://cocodataset.org](https://cocodataset.org/#download)
2. Extract to a directory, e.g.: 

```
/data/coco/
├── images/
│   └── val2017/
│       ├── 000000000139.jpg
│       └── ...
└── annotations/
    └── instances_val2017.json
```

3. Update the annotation file paths in the scripts (default: `/home/almaankhan/data/coco/...`) to match your local paths.

---

## Key Concepts

### Long-Tail Distribution
A statistical property of real-world datasets where a small number of categories appear very frequently ("head"), while most categories are rare ("tail"). This causes standard ERM-trained models to be biased toward head classes.

### Focal Loss (γ, α)
A modified cross-entropy that multiplies the standard loss by `(1 - p_t)^γ`, down-weighting easy examples. The `α` term further balances positive/negative samples.

### Varifocal Loss
An IoU-aware extension of Focal Loss that asymmetrically weights positive samples by their localization quality (IoU with ground truth), making confidence scores more meaningful.

### Weighted Boxes Fusion (WBF)
An ensemble fusion strategy that merges overlapping boxes from multiple detectors by averaging their coordinates weighted by confidence scores, rather than greedily suppressing like NMS.

### Head / Medium / Tail Grouping
Classes are split into three frequency tiers for granular evaluation:
- **Head (14 classes):** person, car, dog, cat, bicycle, boat, bird, chair, cow, horse, sheep, traffic light, stop sign, bench
- **Tail (13 classes):** toothbrush, hair drier, scissors, vase, clock, book, remote, keyboard, mouse, laptop, oven, toaster, sink
- **Medium (53 classes):** all remaining COCO categories

---

## References

1. Lin, T.-Y., et al. *"Focal Loss for Dense Object Detection"*. ICCV 2017. [[arXiv:1708.02002]](https://arxiv.org/abs/1708.02002)
2. Zhang, H., et al. *"VarifocalNet: An IoU-aware Dense Object Detector"*. CVPR 2021. [[arXiv:2008.13367]](https://arxiv.org/abs/2008.13367)
3. Solovyev, R., et al. *"Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models"*. 2020. [[arXiv:1910.13302]](https://arxiv.org/abs/1910.13302)
4. Lin, T.-Y., et al. *"Microsoft COCO: Common Objects in Context"*. ECCV 2014. [[arXiv:1405.0312]](https://arxiv.org/abs/1405.0312)
5. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
6. [TorchVision Detection Models](https://pytorch.org/vision/stable/models.html#object-detection)
