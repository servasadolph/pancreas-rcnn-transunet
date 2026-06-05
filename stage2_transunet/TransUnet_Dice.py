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
# import os    
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torchvision
import torch

# torch.cuda.is_available # cuda가 사용가능하면 True 아니면 False 리턴
# torch.cuda.device_count()

# if torch.cuda.is_available():    
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch

print("PyTorch sees", torch.cuda.device_count(), "GPUs\n")

for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))

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
        mask = (m_arr > 0).astype('float32')

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)
        return image_id, image, mask

# %%
dataset_train = CustomDataset(dataset)
print(len(dataset_train))

# %%
# img = np.transpose(dataset_train[0][1].cpu(), (1, 2, 0))
# plt.imshow(img)
# plt.show()

# %%
# plt.imshow(dataset_train[0][2])
# plt.show()

# %%
# np.unique(dataset_train[0][2])

# %%
datasets = train_val_dataset(dataset_train)
print(len(datasets['train']))
print(len(datasets['val']))
print(len(datasets['test']))

# %%
loader_train = DataLoader(datasets['train'], batch_size=4, shuffle=True)
loader_val = DataLoader(datasets['val'], batch_size=4, shuffle=True)
loader_test = DataLoader(datasets['test'], batch_size=1, shuffle=False)

# %%
# Image.open('./data/pancreas_ok_dataset/train_img/img41_103.png')

# %%
Image.open('./data/pancreas_ok_dataset/train_mask/mask41_103.png')

# %%
print("=== 데이터 진단 ===")
for i, (ids, imgs, masks) in enumerate(loader_train):
    print(ids)
    print(f"배치 {i+1}:")
    print(f"  Images 형태: {imgs.shape}, 범위: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  Masks 형태: {masks.shape}, 범위: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"  Masks 고유값: {torch.unique(masks)}")
    print(f"  Positive 픽셀 비율: {(masks > 0).float().mean():.3f}")
    
    if i >= 2:  # 3개 배치만 확인
        break

# %%
iterator = iter(loader_train)
idx, imgs, masks = next(iterator)
print(imgs.shape, masks.shape)
print(imgs.dtype)

# %% [markdown]
# # 모델 정의

# %%
TRAIN_IMG_SIZE = (512, 512, 3)
VAL_IMG_SIZE = TRAIN_IMG_SIZE
TEST_IMG_SIZE = TRAIN_IMG_SIZE
N_CLASSES = 2
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
NUM_EPOCHS = 100
TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2
TEST_NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 0.01
TRAIN_NUM_WORKERS = 2
DEVICE = 'cuda'

LOAD_MODEL = False
START_EPOCH = 1


# %%
class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        # Softmax 적용
        inputs = torch.softmax(inputs, dim=1)
        
        # target이 [B, H, W] 형태라면
        if len(target.shape) == 3:
            target = self._one_hot_encoder(target)
        # target이 [B, 1, H, W] 형태라면
        elif len(target.shape) == 4 and target.shape[1] == 1:
            target = self._one_hot_encoder(target.squeeze(1))
        
        if weight is None:
            weight = [0.3, 0.7]  # 배경보다 개체에 더 높은 가중치
            
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

class MixedLoss(nn.Module):
    def __init__(self, n_classes=2, alpha=0.5):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(n_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).cuda())
        
    def forward(self, inputs, targets):
        # inputs: [B, 2, H, W] (raw logits)
        # targets: [B, 1, H, W] 또는 [B, H, W]
        
        if len(targets.shape) == 4:
            targets_for_dice = targets.clone()
            targets = targets.squeeze(1)  # CE용: [B, H, W]
        else:
            targets_for_dice = targets.unsqueeze(1)  # Dice용: [B, 1, H, W]
            
        dice = self.dice_loss(inputs, targets_for_dice)
        ce = self.ce_loss(inputs, targets.long())
        return self.alpha * dice + (1 - self.alpha) * ce
    
# 3. test_single_volume 함수 수정 - sigmoid 제거
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # sigmoid 제거, softmax는 모델 내부나 loss에서 처리
                out = torch.argmax(outputs, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            # sigmoid 제거
            out = torch.argmax(outputs, dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    metric_list = []
    for i in range(1, classes):  # 배경 제외하고 계산
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list



# %%
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2688
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../input/project-transunet/project_TransUNet/model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'sigmoid'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../input/project-transunet/project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'sigmoid'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../input/project-transunet/project_TransUNet/model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config



# %%
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# %%
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # Softmax 사용 (sigmoid 대신)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)  # sigmoid -> softmax
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        #print(embeddings.shape) #torch.Size([4, 1024, 768])
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=512, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        
        #print(logits.shape) #torch.Size([4, 2, 512, 512])
        return logits #여기까진 잘 나옴 

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'testing': get_testing(),
}

# %%
zzroot_path = '../data/Synapse/train_npz'
zzdataset = 'Synapse'
zzlist_dir = './lists/lists_Synapse'
zznum_classes = 1
zzmax_iterations = 1000
zzmax_epochs = NUM_EPOCHS
zzbatch_size = TRAIN_BATCH_SIZE
zzn_gpu = 1
zzdeterministic = 1
zzbase_lr = LEARNING_RATE
zzimg_size = TRAIN_IMG_SIZE[0]
zzseed = 1234
zzn_skip = 3
zzvit_name = 'R50-ViT-B_16'
zzvit_patches_size = 16

# %%
from pathlib import Path
base = Path.home() / "runs" / "model"   # 예: ~/runs/model
base.mkdir(parents=True, exist_ok=True)

snapshot_path = str(base)  

# %%
import os, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

# --------------------
# 시드 고정
# --------------------
if not zzdeterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(zzseed)
np.random.seed(zzseed)
torch.manual_seed(zzseed)
torch.cuda.manual_seed_all(zzseed)

# --------------------
# 데이터셋 설정
# --------------------
dataset_name = zzdataset
dataset_config = {
    'Synapse': {
        'root_path': '../input/project-transunet/project_TransUNet/data/Synapse/train_npz',
        'list_dir': '../input/project-transunet/project_TransUNet/TransUNet/lists/lists_Synapse',
        'num_classes': N_CLASSES,
    },
}
zznum_classes = N_CLASSES
zzroot_path = dataset_config[dataset_name]['root_path']
zzlist_dir = dataset_config[dataset_name]['list_dir']

# --------------------
# 스냅샷 경로 안전하게 구성
# --------------------
zzis_pretrain = True
zzexp = f"TU_{dataset_name}{zzimg_size}"
snapshot_path = Path.home() / "runs" / "model" / zzexp / "TU"

if zzis_pretrain:
    snapshot_path = Path(str(snapshot_path) + "_pretrain")

snapshot_path = Path(f"{snapshot_path}_{zzvit_name}")
snapshot_path = Path(f"{snapshot_path}_skip{zzn_skip}")

if zzvit_patches_size != 16:
    snapshot_path = Path(f"{snapshot_path}_vitpatch{zzvit_patches_size}")

if zzmax_iterations != 30000:
    snapshot_path = Path(f"{snapshot_path}_{str(zzmax_iterations)[:2]}k")

if zzmax_epochs != 30:
    snapshot_path = Path(f"{snapshot_path}_epo{zzmax_epochs}")

snapshot_path = Path(f"{snapshot_path}_bs{zzbatch_size}")

if zzbase_lr != 0.01:
    snapshot_path = Path(f"{snapshot_path}_lr{zzbase_lr}")

snapshot_path = Path(f"{snapshot_path}_{zzimg_size}")

if zzseed != 1234:
    snapshot_path = Path(f"{snapshot_path}_s{zzseed}")

# 디렉토리 생성 (권한 문제시 홈으로 폴백)
try:
    snapshot_path.mkdir(parents=True, exist_ok=True)
except PermissionError:
    alt = Path.home() / "runs" / snapshot_path.name
    alt.mkdir(parents=True, exist_ok=True)
    print(f"[경고] {snapshot_path} 접근 불가 → {alt} 로 변경")
    snapshot_path = alt

print("Snapshot path:", snapshot_path)

# --------------------
# 모델 구성
# --------------------
def create_model():
    config_vit = CONFIGS[zzvit_name]
    config_vit.n_classes = 2
    config_vit.n_skip = zzn_skip
    
    if "R50" in zzvit_name:
        config_vit.patches.grid = (
            int(zzimg_size / zzvit_patches_size),
            int(zzimg_size / zzvit_patches_size),
        )
    
    model = VisionTransformer(config_vit, img_size=zzimg_size,
                              num_classes=config_vit.n_classes).cuda()
    
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    
    return model


# %%
def calculate_iou(y_pred, y_true, threshold=0.5):
    """IoU 계산"""
    # y_pred: [B, 2, H, W] (logits)
    # y_true: [B, H, W] 또는 [B, 1, H, W]
    
    if len(y_true.shape) == 4:
        y_true = y_true.squeeze(1)
        
    # Softmax 후 클래스 1 확률
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = y_pred[:, 1]  # 개체 클래스
    
    # 임계값 적용
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    
    iou = intersection / (union + 1e-7)
    return iou.item()

def calculate_dice(y_pred, y_true, threshold=0.5):
    """Dice 계수 계산"""
    if len(y_true.shape) == 4:
        y_true = y_true.squeeze(1)
        
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = y_pred[:, 1]
    
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    
    intersection = (y_pred * y_true).sum()
    dice = (2 * intersection) / (y_pred.sum() + y_true.sum() + 1e-7)
    return dice.item()


# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch_data in pbar:
        # 데이터 추출
        if len(batch_data) == 3:
            _, images, masks = batch_data
        else:
            images, masks = batch_data
        
        images = images.to(device)
        masks = masks.to(device)
        
        # 마스크 전처리
        if len(masks.shape) == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        # 0, 1로 이진화
        masks = (masks > 0).long()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 메트릭 계산
        with torch.no_grad():
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)
        
        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        num_batches += 1
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'Loss': f'{running_loss/num_batches:.4f}',
            'IoU': f'{running_iou/num_batches:.4f}',
            'Dice': f'{running_dice/num_batches:.4f}'
        })
    
    return {
        'loss': running_loss / num_batches,
        'iou': running_iou / num_batches,
        'dice': running_dice / num_batches
    }

# 6. 검증 함수
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Validation")
    
    with torch.no_grad():
        for batch_data in pbar:
            # 데이터 추출
            if len(batch_data) == 3:
                _, images, masks = batch_data
            else:
                images, masks = batch_data
            
            images = images.to(device)
            masks = masks.to(device)
            
            # 마스크 전처리
            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            masks = (masks > 0).long()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 메트릭 계산
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            num_batches += 1
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'Loss': f'{running_loss/num_batches:.4f}',
                'IoU': f'{running_iou/num_batches:.4f}',
                'Dice': f'{running_dice/num_batches:.4f}'
            })
    
    return {
        'loss': running_loss / num_batches,
        'iou': running_iou / num_batches,
        'dice': running_dice / num_batches
    }

# 7. 전체 설정
# def setup_training():
#     # 모델 생성
#     model = create_model()
    
#     # Loss와 optimizer
#     criterion = DiceLoss(n_classes=2)
#     #criterion = MixedLoss(n_classes=2, alpha=0.7)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
#     #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer, T_0=20, T_mult=2, eta_min=1e-6
#     )
    
#     return model, criterion, optimizer, scheduler

# 8. 모델 출력 확인
#model, criterion, optimizer, scheduler = setup_training()

print("모델 출력 확인:")
with torch.no_grad():
    dummy_input = torch.randn(2, 3, 512, 512).cuda()
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # [2, 2, 512, 512]이어야 함

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_csv = save_dir / "metrics_log100_after.csv"

    best_val_iou = 0.0

    # ✅ 에포크 로그 저장용 버퍼
    history = {
        "epoch": [],
        "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss": [],   "val_iou": [],   "val_dice": [],
        "lr": []
    }

    for epoch in range(100,num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # 훈련
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 검증
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)

        # 스케줄러 업데이트
        scheduler.step()

        # 현재 LR
        cur_lr = optimizer.param_groups[0]['lr']

        # 결과 출력
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"Learning Rate: {cur_lr:.2e}")

        # ✅ 히스토리 버퍼에 저장
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["lr"].append(cur_lr)

        # ✅ CSV로 즉시 기록 (중간 종료 대비)
        pd.DataFrame(history).to_csv(log_csv, index=False)

        # Best 모델 저장
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), save_dir / 'DICE_HD_best_model.pth')
            print(f"New best model saved! IoU: {best_val_iou:.4f}")

        # 에포크마다 저장(예: 20 에포크 단위)
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), save_dir / f'DICE_HD_epoch_{epoch+1}.pth')

    print(f"\n📄 Metrics log saved to: {log_csv}")
    return history



# %%
from tqdm import tqdm
print("훈련 시작...")

#dice+SGD
train_model( 
    model=model,
    train_loader=loader_train,
    val_loader=loader_val,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=NUM_EPOCHS,
    save_path="./weights"
)

#MixedLoss(dice+CrossEntropyLoss)+ADAM
# train_model(
#     model=model,
#     train_loader=loader_train,
#     val_loader=loader_val,
#     criterion=criterion,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     num_epochs=NUM_EPOCHS,
#     save_path=snapshot_path
# )

# %%
#50epoch 더 돌려보기
train_model( 
    model=model,
    train_loader=loader_train,
    val_loader=loader_val,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=150,
    save_path="./weights"
)



# %%
import torch
from collections import OrderedDict
model = create_model()
#ckpt = torch.load("./weights/best_model.pth")
ckpt = torch.load("./weights/DICE_HD_best_model.pth")

# 이미 DataParallel(model) 상태라면, 키에 module. 접두사 추가
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
else:
    sd = ckpt

# 키 확인 후 접두사 추가
first_key = next(iter(sd))
if not first_key.startswith("module."):
    new_sd = OrderedDict((f"module.{k}", v) for k, v in sd.items())
else:
    new_sd = sd

# num_batches_tracked 등 버퍼 차이를 무시하고 싶으면 strict=False
model.load_state_dict(new_sd, strict=False)
model.eval()

# %%
import torch
import numpy as np
from typing import Optional

def _to_tensor(x, like: Optional[torch.Tensor] = None, dtype=torch.float32) -> torch.Tensor:
    """np.ndarray나 torch.Tensor를 받아 torch.Tensor로 통일. like 텐서가 있으면 같은 device로 이동."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be numpy.ndarray or torch.Tensor")
    x = x.to(dtype=dtype)
    if like is not None:
        x = x.to(device=like.device)
    return x

def _prep_binary(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    mask: (B,H,W) 또는 (B,1,H,W) 또는 (B,C,H,W)
    float/logit/long 허용. 반환은 bool (B,H,W)
    """
    # (B,1,H,W) -> (B,H,W)
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() == 4 and mask.size(1) > 1:
        # 멀티클래스라면 채널 argmax로 이진화
        mask = mask.argmax(dim=1)

    # float면 threshold, 정수/불리언이면 >0
    if mask.dtype.is_floating_point:
        bin_mask = mask > threshold
    else:
        bin_mask = mask > 0
    return bin_mask

def dice_coeff(y_true, y_pred, eps: float = 1e-7, threshold: float = 0.5) -> torch.Tensor:
    # 타입/디바이스 맞추기
    y_true = _to_tensor(y_true, dtype=torch.float32)
    y_pred = _to_tensor(y_pred, like=y_true, dtype=torch.float32)

    gt = _prep_binary(y_true, threshold)
    pr = _prep_binary(y_pred, threshold)

    B = gt.size(0)
    gt = gt.reshape(B, -1)
    pr = pr.reshape(B, -1)

    inter = (gt & pr).sum(dim=1).float()
    gt_sum = gt.sum(dim=1).float()
    pr_sum = pr.sum(dim=1).float()
    denom = gt_sum + pr_sum

    both_empty = (denom == 0)
    dice = (2 * inter + eps) / (denom + eps)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.mean()

def iou_coeff(y_true, y_pred, eps: float = 1e-7, threshold: float = 0.5) -> torch.Tensor:
    y_true = _to_tensor(y_true, dtype=torch.float32)
    y_pred = _to_tensor(y_pred, like=y_true, dtype=torch.float32)

    gt = _prep_binary(y_true, threshold)
    pr = _prep_binary(y_pred, threshold)

    B = gt.size(0)
    gt = gt.reshape(B, -1)
    pr = pr.reshape(B, -1)

    inter = (gt & pr).sum(dim=1).float()
    union = (gt | pr).sum(dim=1).float()

    both_empty = (union == 0)
    iou = (inter + eps) / (union + eps)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    return iou.mean()



# %%
from skimage.morphology import binary_erosion, square
import matplotlib.patches as mpatches

# 결과 시각화
iterator = iter(loader_test)
idx, imgs, masks = next(iterator)           # imgs: [B,3,H,W], masks: [B,H,W] or [B,1,H,W]
preds = model(imgs.to(DEVICE))              # preds: [B,C,H,W] (logit or prob)

print(imgs.shape)
print(masks.shape)
print(preds.shape)

a = preds.detach().cpu().numpy()            # [B,C,H,W]
B = a.shape[0]
fg_ch = 1 if a.shape[1] > 1 else 0          # C>1이면 1번 채널을 foreground로 가정, 단일채널이면 0

rows = min(4, B)
fig, axs = plt.subplots(nrows=rows, ncols=2, figsize=(15, 30))
if rows == 1:
    axs = np.array([axs])                   # 인덱싱 통일

selem = square(5)                           # 2D footprint
iou_batch_loss = []

for i in range(rows):
    # --- 원본 이미지 준비 (0~1 스케일 보정) ---
    img = imgs[i].cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    if img.max() > 1.0:
        img = img / 255.0

    # --- GT 마스크: torch -> numpy, 이진화 ---
    m = masks[i].detach().cpu()
    if m.ndim == 3 and m.shape[0] == 1:     # [1,H,W]인 경우 squeeze
        m = m.squeeze(0)
    mask_t = (m > 0).to(torch.uint8)        # [H,W], {0,1}
    mask_np = mask_t.numpy().astype(np.uint8)

    # --- 좌측: 원본 + GT(초록) 오버레이 ---
    gt_overlay = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    gt_overlay[mask_np == 1] = [0, 1, 0]    # GT=초록
    left_viz = np.clip(0.5 * img + 0.5 * gt_overlay, 0, 1)
    axs[i, 0].imshow(left_viz)
    axs[i, 0].set_title('RAW + GT(green)')
    axs[i, 0].axis('off')

    # --- 예측 마스크 (2D) 생성 ---
    pred_prob = a[i, fg_ch]                 # (H,W)
    pred_bin = (pred_prob >= 0.75)          # bool 2D

    # --- 형태학적 침식 ---
    eroded = binary_erosion(pred_bin, footprint=selem)  # bool 2D
    pred = eroded.astype(np.uint8)                      # 0/1

    # --- TP/FP/FN 분해 색상 ---
    tp = (mask_np == 1) & (pred == 1)   # 노랑
    fp = (mask_np == 0) & (pred == 1)   # 빨강
    fn = (mask_np == 1) & (pred == 0)   # 초록

    overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)
    overlay[tp] = [1, 1, 0]   # TP = 노랑
    overlay[fp] = [1, 0, 0]   # FP = 빨강
    overlay[fn] = [0, 1, 0]   # FN = 초록
    right_viz = np.clip(0.5 * img + 0.5 * overlay, 0, 1)

    axs[i, 1].imshow(right_viz)
    axs[i, 1].set_title('GT(green) & Pred(red) → Union(yellow(TP))')
    axs[i, 1].axis('off')

    # 범례
    tp_patch = mpatches.Patch(color='yellow', label='TP (Union)')
    fp_patch = mpatches.Patch(color='red',    label='FP (Pred)')
    fn_patch = mpatches.Patch(color='green',  label='FN (Ground truth)')
    axs[i, 1].legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right')

    # --- metric 계산 (torch 텐서로 맞춤) ---
    pred_t = torch.from_numpy(pred).to(mask_t.device).type_as(mask_t)  # [H,W], 0/1
    dc = dice_coeff(mask_t, pred_t)
    io = iou_coeff(mask_t, pred_t)

    print(f"[{i}] IoU={io.item() if hasattr(io,'item') else io:.4f}, Dice={dc.item() if hasattr(dc,'item') else dc:.4f}")
    iou_batch_loss.append(io.item() if hasattr(io, "item") else float(io))

print("Mean IoU:", np.mean(iou_batch_loss))
plt.tight_layout()
plt.show()


# %%

# %%
#DICE_HD_best_model
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from skimage.morphology import binary_erosion, square
import hashlib, re
import matplotlib.patches as mpatches

out_dir = Path("./outputs/dice_test_vis")
raw_dir  = out_dir / "raw_gt"
pred_dir = out_dir / "pred_overlay"
raw_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

dice_list, iou_list, ids_list, safe_ids = [], [], [], []
selem = square(5)

def safe_basename(x: str) -> str:
    s = str(x).replace("\\", "/")
    p = Path(s)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)
    h = hashlib.md5(s.encode()).hexdigest()[:6]
    return f"{stem}_{h}"

with torch.no_grad():
    for image_ids, imgs, masks in loader_test:   # batch_size=1
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        a = preds.detach().cpu().numpy()
        fg_ch = 1 if a.shape[1] > 1 else 0

        # ----- 원본 -----
        img = imgs[0].cpu().numpy().transpose(1, 2, 0)
        if img.max() > 1.0:
            img = img / 255.0

        # ----- GT -----
        m = masks[0].cpu()
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        mask_t  = (m > 0).to(torch.uint8)
        mask_np = mask_t.numpy().astype(np.uint8)

        # ----- 예측 -----
        pred_prob = a[0, fg_ch]
        pred_bin  = (pred_prob >= 0.75)
        eroded    = binary_erosion(pred_bin, footprint=selem)
        pred      = eroded.astype(np.uint8)

        # ----- 색상 오버레이 -----
        tp = (mask_np == 1) & (pred == 1)
        fp = (mask_np == 0) & (pred == 1)
        fn = (mask_np == 1) & (pred == 0)
        overlay = np.zeros((*mask_np.shape, 3), dtype=np.float32)
        overlay[tp] = [1, 1, 0]
        overlay[fp] = [1, 0, 0]
        overlay[fn] = [0, 1, 0]

        # RAW+GT
        gt_overlay = np.zeros_like(overlay)
        gt_overlay[mask_np == 1] = [0, 1, 0]
        left_viz  = np.clip(0.5 * img + 0.5 * gt_overlay, 0, 1)
        right_viz = np.clip(0.5 * img + 0.5 * overlay, 0, 1)

        # ----- metric -----
        pred_t = torch.from_numpy(pred).to(mask_t.device).type_as(mask_t)
        dc  = dice_coeff(mask_t.unsqueeze(0), pred_t.unsqueeze(0)).item()
        io  = iou_coeff(mask_t.unsqueeze(0),  pred_t.unsqueeze(0)).item()

        orig_id  = str(image_ids[0])
        title_id = Path(orig_id.replace("\\", "/")).name
        file_id  = safe_basename(orig_id)

        dice_list.append(dc)
        iou_list.append(io)
        ids_list.append(orig_id)
        safe_ids.append(file_id)

        # ----- 저장 -----
        # RAW+GT
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(left_viz); ax1.set_title(f"{title_id} (RAW+GT)"); ax1.axis("off")
        fig1.savefig(raw_dir / f"{file_id}.png", dpi=150)
        plt.close(fig1)

        # Pred overlay
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(right_viz); ax2.set_title(f"{title_id} (GT/Pred/TP)"); ax2.axis("off")
        tp_patch = mpatches.Patch(color='yellow', label='TP')
        fp_patch = mpatches.Patch(color='red',    label='FP')
        fn_patch = mpatches.Patch(color='green',  label='FN')
        ax2.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right')
        fig2.savefig(pred_dir / f"{file_id}.png", dpi=150)
        plt.close(fig2)

# ----- 전체 평균 -----
mean_dice = float(np.mean(dice_list))
mean_iou  = float(np.mean(iou_list))
print(f"[전체 평균] Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")

# ----- CSV 저장 -----
df = pd.DataFrame({
    "image_id": ids_list,
    "safe_id":  safe_ids,
    "dice":     dice_list,
    "iou":      iou_list
})
df.loc[len(df)] = ["mean", "mean", mean_dice, mean_iou]
df.to_csv(out_dir / "metrics_test.csv", index=False)


# %%
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import pandas as pd
# from skimage.morphology import binary_erosion, square
# import hashlib, re
# import matplotlib.patches as mpatches

# out_dir = Path("./outputs/dice_test_vis")
# raw_dir  = out_dir / "raw_gt"
# pred_dir = out_dir / "pred_overlay"
# raw_dir.mkdir(parents=True, exist_ok=True)
# pred_dir.mkdir(parents=True, exist_ok=True)

# dice_list, iou_list, ids_list, safe_ids = [], [], [], []
# selem = square(5)

# def safe_basename(x: str) -> str:
#     s = str(x).replace("\\", "/")
#     p = Path(s)
#     stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)
#     h = hashlib.md5(s.encode()).hexdigest()[:6]
#     return f"{stem}_{h}"

# with torch.no_grad():
#     for image_ids, imgs, masks in loader_test:   # batch_size=1
#         imgs  = imgs.to(DEVICE)
#         masks = masks.to(DEVICE)

#         preds = model(imgs)
#         a = preds.detach().cpu().numpy()
#         fg_ch = 1 if a.shape[1] > 1 else 0

#         # ----- 원본 -----
#         img = imgs[0].cpu().numpy().transpose(1, 2, 0)
#         if img.max() > 1.0:
#             img = img / 255.0

#         # ----- GT -----
#         m = masks[0].cpu()
#         if m.ndim == 3 and m.shape[0] == 1:
#             m = m.squeeze(0)
#         mask_t  = (m > 0).to(torch.uint8)
#         mask_np = mask_t.numpy().astype(np.uint8)

#         # ----- 예측 -----
#         pred_prob = a[0, fg_ch]
#         pred_bin  = (pred_prob >= 0.75)
#         eroded    = binary_erosion(pred_bin, footprint=selem)
#         pred      = eroded.astype(np.uint8)

#         # ----- 색상 오버레이 -----
#         tp = (mask_np == 1) & (pred == 1)
#         fp = (mask_np == 0) & (pred == 1)
#         fn = (mask_np == 1) & (pred == 0)
#         overlay = np.zeros((*mask_np.shape, 3), dtype=np.float32)
#         overlay[tp] = [1, 1, 0]
#         overlay[fp] = [1, 0, 0]
#         overlay[fn] = [0, 1, 0]

#         # RAW+GT
#         gt_overlay = np.zeros_like(overlay)
#         gt_overlay[mask_np == 1] = [0, 1, 0]
#         left_viz  = np.clip(0.5 * img + 0.5 * gt_overlay, 0, 1)
#         right_viz = np.clip(0.5 * img + 0.5 * overlay, 0, 1)

#         # ----- metric -----
#         pred_t = torch.from_numpy(pred).to(mask_t.device).type_as(mask_t)
#         dc  = dice_coeff(mask_t.unsqueeze(0), pred_t.unsqueeze(0)).item()
#         io  = iou_coeff(mask_t.unsqueeze(0),  pred_t.unsqueeze(0)).item()

#         orig_id  = str(image_ids[0])
#         title_id = Path(orig_id.replace("\\", "/")).name
#         file_id  = safe_basename(orig_id)

#         dice_list.append(dc)
#         iou_list.append(io)
#         ids_list.append(orig_id)
#         safe_ids.append(file_id)

#         # ----- 저장 -----
#         # RAW+GT
#         fig1, ax1 = plt.subplots(figsize=(5, 5))
#         ax1.imshow(left_viz); ax1.set_title(f"{title_id} (RAW+GT)"); ax1.axis("off")
#         fig1.savefig(raw_dir / f"{file_id}.png", dpi=150)
#         plt.close(fig1)

#         # Pred overlay
#         fig2, ax2 = plt.subplots(figsize=(5, 5))
#         ax2.imshow(right_viz); ax2.set_title(f"{title_id} (GT/Pred/TP)"); ax2.axis("off")
#         tp_patch = mpatches.Patch(color='yellow', label='TP')
#         fp_patch = mpatches.Patch(color='red',    label='FP')
#         fn_patch = mpatches.Patch(color='green',  label='FN')
#         ax2.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right')
#         #fig2.savefig(pred_dir / f"{file_id}.png", dpi=150)
#         plt.close(fig2)

# # ----- 전체 평균 -----
# mean_dice = float(np.mean(dice_list))
# mean_iou  = float(np.mean(iou_list))
# print(f"[전체 평균] Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")

# # ----- CSV 저장 -----
# df = pd.DataFrame({
#     "image_id": ids_list,
#     "safe_id":  safe_ids,
#     "dice":     dice_list,
#     "iou":      iou_list
# })
# df.loc[len(df)] = ["mean", "mean", mean_dice, mean_iou]
# #df.to_csv(out_dir / "metrics_test.csv", index=False)

