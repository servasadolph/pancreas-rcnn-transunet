<h1 align="center">Pancreas Segmentation with a Two-Stage Pipeline of Faster R-CNN and TransUNet</h1>

<p align="center">
  A deep learning framework for automatic pancreas segmentation in abdominal CT.
  The pipeline first localizes the pancreas with Faster R-CNN, then segments it with a
  TransUNet network trained under a Dice–Hausdorff Distance loss.
</p>

<p align="center">
  <b>Yunjung Hong</b> &nbsp;·&nbsp; <b>Servas Adolph Tarimo</b> &nbsp;·&nbsp; <b>Jiyoung Woo</b><br>
  <em>Applied Sciences</em> (MDPI) — to appear
</p>

<!-- HIDDEN UNTIL OFFICIAL PUBLICATION (reveal by deleting this comment block)
Manuscript ID: applsci-4222871
Official article link: <add DOI / URL here>
-->

---

## Overview

The pancreas is among the most difficult abdominal organs to segment automatically: it is small,
varies widely in shape and position, and has low contrast against neighbouring tissue. This project
addresses these challenges with a **two-stage pipeline** that separates *where the pancreas is*
from *what its exact boundary is*.

```
Full 512x512 CT slice  -->  Stage 1: Faster R-CNN  -->  predicted box  -->  crop
                                                                             |
                                           mask  <--  Stage 2: TransUNet  <--+
```

- **Stage 1 — Localization.** A COCO-pretrained **Faster R-CNN (ResNet-50 FPN)** is fine-tuned to
  detect the pancreas and return a bounding box. The box is enlarged slightly and used to crop the
  region of interest, removing most irrelevant background.
- **Stage 2 — Segmentation.** A **TransUNet** segments the pancreas within the cropped region. It is
  trained with a **Dice–Hausdorff Distance (DHD) loss**, `L = L_Dice + alpha * L_surface`, which
  combines region overlap (Dice) with boundary accuracy (Hausdorff distance).

<p align="center">
  <img src="figures/figure01.png" alt="Overall framework: data preparation, two-stage Faster R-CNN and TransUNet segmentation, and evaluation" width="100%">
  <br><em>Figure 1. Overall framework: data preparation, two-stage localization and segmentation, and evaluation.</em>
</p>

---

## Results

<!-- HIDDEN UNTIL OFFICIAL PUBLICATION (reveal by deleting this comment block)

### NIH Pancreas-CT dataset

| Model | Mean DSC (%) | Std (DSC) | Precision (%) | Recall (%) |
|---|:---:|:---:|:---:|:---:|
| Faster R-CNN + TransUNet (DHD) | 88.98 | 17.10 | 91.0 | 94.4 |
| Faster R-CNN + TransUNet (DSC) | 87.91 | 16.60 | 92.5 | 93.0 |
| U-Net (baseline)               | 83.6  | 27.1 | 91.1 | 81.6 |

### Cross-dataset evaluation — BTCV (30 volumes)

| Setting | Mean DSC (%) | Std (DSC) | Precision (%) | Recall (%) |
|---|:---:|:---:|:---:|:---:|
| Zero-shot (NIH model, no fine-tuning) | 62.96 | 10.51 | 63.64 | 66.44 |
| Fully automatic (5-fold CV)           | 66.50 | 8.55  | 63.82 | 79.29 |

-->

Quantitative results will be added here after the article is officially published.

<p align="center">
  <img src="figures/fig6.png" alt="Qualitative comparison of segmentation outputs from each model" width="80%">
  <br><em>Figure 2. Qualitative comparison of segmentation outputs. From left to right: CT image with
  ground truth (green), and predictions from the DHD, DSC, and U-Net models.</em>
</p>

---

## Repository structure

```
pancreas-rcnn-transunet/
├── inference_time.py             # Shared model definition (create_model) and timing utilities
│
├── stage1_rcnn/                  # Stage 1 — Faster R-CNN localization
│   ├── step1_rcnn_finetune.py        train the detector
│   └── step2_rcnn_predict.py         run the detector to produce boxes and crops
│
├── stage2_transunet/             # Stage 2 — TransUNet segmentation
│   ├── TransUNet_DiceHD.py           proposed model (Dice-Hausdorff loss)
│   ├── TransUnet_Dice.py             Dice-loss variant
│   └── UNet_.py                      U-Net baseline
│
├── data/                         # Data loading and preprocessing
│   ├── btcv_load_preprocess.py       convert BTCV volumes to training slices
│   └── datacheck.py                  dataset sanity checks
│
├── evaluate_btcv/                # Cross-dataset evaluation on BTCV
│   ├── btcv_evaluate_v9.py           zero-shot evaluation
│   └── btcv_5fold_cv.py              fully automatic 5-fold cross-validation
│
├── figures/                      # Paper figures
├── weights/                      # Place trained checkpoints here (not included)
├── outputs/                      # Generated results and checkpoints (git-ignored)
├── requirements.txt
└── LICENSE
```

> Run every script from the repository root, for example `python evaluate_btcv/btcv_5fold_cv.py`.
> All scripts import `create_model` from the root-level `inference_time.py` and resolve the
> `data/`, `weights/`, and `outputs/` folders relative to the repository root.

---

## Installation

```bash
git clone https://github.com/servasadolph/pancreas-rcnn-transunet.git
cd pancreas-rcnn-transunet

python -m venv .venv && source .venv/bin/activate    # or use conda
pip install -r requirements.txt
```

Implemented and tested with **PyTorch 2.3.0** (torchvision 0.18.0) on Python 3.8+.
Models were trained on two NVIDIA RTX A4000 GPUs using `torch.nn.DataParallel`.

---

## Datasets

This repository contains source code only. The CT datasets used in this study are publicly available:

| Dataset | Use | Source |
|---|---|---|
| NIH Pancreas-CT | Main training and evaluation | [kaggle.com/datasets/tahsin/pancreasct-dataset](https://www.kaggle.com/datasets/tahsin/pancreasct-dataset) (originally from The Cancer Imaging Archive) |
| BTCV (Beyond the Cranial Vault) | Cross-dataset evaluation | [kaggle.com/datasets/lssz1275/abdomen](https://www.kaggle.com/datasets/lssz1275/abdomen) (MICCAI 2015 Multi-Atlas Labeling) |

Expected layout after download:

```
data/
├── btcv_datasets/
│   ├── imagesTr/        # BTCV CT volumes (img0001.nii.gz, ...)
│   └── labelsTr/        # BTCV labels (pancreas label == 11)
├── pancreas_ok_dataset/ # NIH cropped slices for Stage-2 training/evaluation
└── data_path_result.csv # NIH image/mask index used by Stage-2 training
```

---

## Pretrained weights

The NIH-pretrained TransUNet checkpoint `weights/DICE_HD_best_model_100after.pth` is required at
run time for the BTCV evaluations and is reused by Stage 2. Because of its size it is not included
in the repository; place it in the `weights/` folder before running the evaluation scripts.

<!-- HIDDEN UNTIL OFFICIAL PUBLICATION (reveal by deleting this comment block)
Download link for trained weights: <add release / cloud link here>
-->

---

## Usage

**Stage 1 — Faster R-CNN detector**

```bash
python stage1_rcnn/step1_rcnn_finetune.py     # trains the detector
python stage1_rcnn/step2_rcnn_predict.py      # produces predicted boxes and crops
```

**Stage 2 — TransUNet training**

```bash
python stage2_transunet/TransUNet_DiceHD.py   # proposed model (Dice-Hausdorff loss)
python stage2_transunet/TransUnet_Dice.py     # Dice-loss variant
python stage2_transunet/UNet_.py              # U-Net baseline
```

**Cross-dataset evaluation on BTCV**

```bash
python evaluate_btcv/btcv_evaluate_v9.py      # zero-shot evaluation
python evaluate_btcv/btcv_5fold_cv.py         # fully automatic 5-fold cross-validation
```

---

## Citation

<!-- HIDDEN UNTIL OFFICIAL PUBLICATION (reveal and complete with the final volume/DOI)

```bibtex
@article{hong2026pancreas,
  title   = {Pancreas Segmentation with a Two-Stage of R-CNN and TransUNet},
  author  = {Hong, Yunjung and Tarimo, Servas Adolph and Woo, Jiyoung},
  journal = {Applied Sciences},
  year    = {2026},
  publisher = {MDPI}
}
```
-->

Citation details will be added once the article is officially published.

---

## License

Released under the [MIT License](LICENSE).
