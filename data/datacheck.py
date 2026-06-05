# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: medseg (cu110, py38)
#     language: python
#     name: medseg_cu110
# ---

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch

print("PyTorch sees", torch.cuda.device_count(), "GPUs\n")

for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))

# %%
import torchvision
import torch

# %%
# (removed Python-2 `from __future__` imports — no-ops on Python 3, and invalid
#  mid-file once exported from the original Jupyter notebook)

import pandas as pd
import numpy as np
import random
import math

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pydicom 
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import nibabel as nib

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.data import Subset
from torch.utils.data import DataLoader
import ml_collections

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import logging
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp #-> 모듈을 못찾을때는 구체적으로 더 타고 들어가는 거면 찾을 수도 있음

torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True

from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk

import copy

from torch.nn import CrossEntropyLoss, Dropout, Sigmoid, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from skimage.morphology import binary_erosion, square
from scipy.spatial.distance import directed_hausdorff

# %%
dataset = pd.read_csv("./data/data_path_result.csv")
dataset

# %%
dataset = dataset.reset_index(drop=True)
dataset


# %%
# 데이터셋 분리
def train_val_dataset(dataset, val_split=0.1):
    data_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1)
    train_idx, val_idx = train_test_split(list(data_idx), test_size=0.2)
    
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    
    return datasets


# %%
from pathlib import Path

class CustomDataset(object):
    def __init__(self, data, transforms=None):
        super().__init__()
        self.image_ids = list(data["train_img"])
        # 윈도우 백슬래시를 슬래시로 교체 + 절대경로화
        self.imgs  = [str(Path(str(p).replace("\\", "/")).resolve()) for p in data['train_img']]
        self.masks = [str(Path(str(p).replace("\\", "/")).resolve()) for p in data['train_mask']]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path  = self.imgs[idx]
        mask_path = self.masks[idx]


        # 이미지
        image = Image.open(img_path).convert('RGB')
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(img_arr, (2, 0, 1)).astype('float32')

        # 마스크
        m = Image.open(mask_path)
        m_arr = np.asarray(m).copy()
        m_arr[m_arr < 8] = 0
        mask = (m_arr > 0).astype('float32')  # [H, W]
        mask = np.expand_dims(mask, axis=0)   # [1, H, W]

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)

        return image_id, image, mask

# %%
dataset_train = CustomDataset(dataset)
print(len(dataset_train))

# %%
datasets = train_val_dataset(dataset_train)
print(len(datasets['train']))
print(len(datasets['val']))
print(len(datasets['test']))

# %%
loader_train = DataLoader(datasets['train'], batch_size=4, shuffle=True)
loader_val = DataLoader(datasets['val'], batch_size=4, shuffle=True)
loader_test = DataLoader(datasets['test'], batch_size=4, shuffle=True)

# %%
Image.open('./data/pancreas_ok_dataset/train_img/img41_103.png')

# %%
print("=== 데이터 진단 ===")
for i, (ids, imgs, masks) in enumerate(loader_train):
    #print(ids)
    print(f"배치 {i+1}:")
    print(f"  Images 형태: {imgs.shape}, 범위: [{imgs.min():.3f}, {imgs.max():.3f}]")
    img = imgs[0]            # (C,H,W)
    img = img.permute(1,2,0) # (H,W,C)
    img = img.numpy()

    # 값이 [0,1] 범위면 그대로, [0,255] 범위면 0~1로 normalize
    if img.max() > 1.0:
        img = img / 255.0

    plt.imshow(img)
    plt.title(f"Batch {i+1} - id={ids[0]}")
    plt.axis("off")
    plt.show()
    
    print(f"  Masks 형태: {masks.shape}, 범위: [{masks.min():.3f}, {masks.max():.3f}]")
    mask = masks[0].numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # (1,H,W) → (H,W)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.show()
    print(f"  Masks 고유값: {torch.unique(masks)}")
    print(f"  Positive 픽셀 비율: {(masks > 0).float().mean():.3f}")
    
    if i >= 2:  # 3개 배치만 확인
        break

# %%
iterator = iter(loader_train)
idx, imgs, masks = next(iterator)
print(imgs.shape, masks.shape)
print(imgs.dtype)
