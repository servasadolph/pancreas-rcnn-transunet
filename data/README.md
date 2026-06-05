# Data

This repository ships **code only**. Download the datasets from their original
sources (data-use agreements apply) and arrange them as below.

```
data/
├── btcv_datasets/
│   ├── imagesTr/        # BTCV CT volumes (img0001.nii.gz …)
│   └── labelsTr/        # BTCV labels (pancreas label == 11)
├── pancreas_ok_dataset/ # NIH cropped slices (train/eval PNGs)
└── data_path_result.csv # NIH image/mask index for Stage-2 training
```

| Dataset | Source |
|---|---|
| NIH Pancreas-CT | The Cancer Imaging Archive (TCIA) |
| BTCV | Synapse / MICCAI 2015 Multi-Atlas Labeling (Beyond the Cranial Vault) |

The image/label files themselves are git-ignored.
