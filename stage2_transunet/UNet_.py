# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (panc)
#     language: python
#     name: panc
# ---

# %%
import torchvision
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

print("PyTorch sees", torch.cuda.device_count(), "GPUs\n")

for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))

# %%
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, transforms=None):
        super().__init__()
        self.image_ids = list(data_df["train_img"])
        self.imgs  = [str(Path(str(p).replace("\\", "/")).resolve()) for p in data_df["train_img"]]
        self.masks = [str(Path(str(p).replace("\\", "/")).resolve()) for p in data_df["train_mask"]]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path  = self.imgs[idx]
        mask_path = self.masks[idx]

        # 이미지
        image = Image.open(img_path).convert("RGB")
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(img_arr, (2, 0, 1)).astype("float32")  # [C,H,W]

        # 마스크
        m = Image.open(mask_path)
        m_arr = np.asarray(m).copy()
        m_arr[m_arr < 8] = 0
        mask = (m_arr > 0).astype("float32")  # [H,W] 0/1

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)

        return image_id, image, mask


def train_val_dataset_fixed(dataset: Dataset, seed: int = 42):
    idx_all = list(range(len(dataset)))

    data_idx, test_idx = train_test_split(
        idx_all, test_size=0.1, random_state=seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        list(data_idx), test_size=0.2, random_state=seed, shuffle=True
    )

    return {
        "train": Subset(dataset, train_idx),
        "val":   Subset(dataset, val_idx),
        "test":  Subset(dataset, test_idx),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_ch=64):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)  
        return out


    
def calculate_precision_recall(outputs, masks, eps=1e-7):
    preds = torch.argmax(outputs, dim=1) 

    B = masks.size(0)
    preds = preds.reshape(B, -1)
    masks = masks.reshape(B, -1)

    tp = ((preds == 1) & (masks == 1)).sum(dim=1).float()
    fp = ((preds == 1) & (masks == 0)).sum(dim=1).float()
    fn = ((preds == 0) & (masks == 1)).sum(dim=1).float()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

    gt_sum = (masks == 1).sum(dim=1).float()
    pr_sum = (preds == 1).sum(dim=1).float()
    both_empty = (gt_sum == 0) & (pr_sum == 0)

    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall    = torch.where(both_empty, torch.ones_like(recall), recall)

    return precision.mean().item(), recall.mean().item()

def calculate_iou(outputs, masks, eps=1e-7):
    preds = torch.argmax(outputs, dim=1)  
    preds = preds.view(-1)
    masks = masks.view(-1)

    inter = ((preds == 1) & (masks == 1)).sum().float()
    union = ((preds == 1) | (masks == 1)).sum().float()
    return (inter / (union + eps)).item()


def calculate_dice(outputs, masks, eps=1e-7):
    preds = torch.argmax(outputs, dim=1) 
    preds = preds.view(-1)
    masks = masks.view(-1)

    inter = ((preds == 1) & (masks == 1)).sum().float()
    p_sum = (preds == 1).sum().float()
    m_sum = (masks == 1).sum().float()
    return ((2 * inter) / (p_sum + m_sum + eps)).item()

def calculate_precision_recall(outputs, masks, eps=1e-7):
    preds = torch.argmax(outputs, dim=1) 

    preds = preds.view(-1)
    masks = masks.view(-1)

    tp = ((preds == 1) & (masks == 1)).sum().float()
    fp = ((preds == 1) & (masks == 0)).sum().float()
    fn = ((preds == 0) & (masks == 1)).sum().float()

    precision = (tp / (tp + fp + eps)).item()
    recall    = (tp / (tp + fn + eps)).item()

    return precision, recall


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_iou, running_dice = 0.0, 0.0, 0.0
    running_precision = 0.0
    running_recall = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch_data in pbar:
        if len(batch_data) == 3:
            _, images, masks = batch_data
        else:
            images, masks = batch_data

        images = images.to(device)
        masks = masks.to(device)

        if len(masks.shape) == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [B,1,H,W] -> [B,H,W]

        masks = (masks > 0).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = dice_coeff(masks, outputs, threshold=0.5).item()
            iou  = iou_coeff(masks, outputs, threshold=0.5).item()
            precision, recall = precision_recall_coeff(masks, outputs, threshold=0.5)
            precision = precision.item()
            recall = recall.item()

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        running_precision += precision
        running_recall += recall
        num_batches += 1

        pbar.set_postfix({
            "Loss": f"{running_loss/num_batches:.4f}",
            "IoU":  f"{running_iou/num_batches:.4f}",
            "Dice": f"{running_dice/num_batches:.4f}",
            "Prec": f"{running_precision/num_batches:.4f}",
            "Rec":  f"{running_recall/num_batches:.4f}",
        })

    return {
        "loss": running_loss / num_batches,
        "iou": running_iou / num_batches,
        "dice": running_dice / num_batches,
        "precision": running_precision / num_batches,
        "recall": running_recall / num_batches
    }


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, running_iou, running_dice = 0.0, 0.0, 0.0
    running_precision = 0.0
    running_recall = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for batch_data in pbar:
            if len(batch_data) == 3:
                _, images, masks = batch_data
            else:
                images, masks = batch_data

            images = images.to(device)
            masks = masks.to(device)

            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            masks = (masks > 0).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            dice = dice_coeff(masks, outputs, threshold=0.5).item()
            iou  = iou_coeff(masks, outputs, threshold=0.5).item()
            precision, recall = precision_recall_coeff(masks, outputs, threshold=0.5)
            precision = precision.item()
            recall = recall.item()

            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            running_precision += precision
            running_recall += recall
            num_batches += 1

            pbar.set_postfix({
                "Loss": f"{running_loss/num_batches:.4f}",
                "IoU":  f"{running_iou/num_batches:.4f}",
                "Dice": f"{running_dice/num_batches:.4f}",
                "Prec": f"{running_precision/num_batches:.4f}",
                "Rec":  f"{running_recall/num_batches:.4f}",
            })

    return {
        "loss": running_loss / num_batches,
        "iou": running_iou / num_batches,
        "dice": running_dice / num_batches,
        "precision": running_precision / num_batches,
        "recall": running_recall / num_batches
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, save_path, start_epoch=0):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_csv = save_dir / "metrics_log.csv"

    best_val_iou = 0.0

    history = {
        "epoch": [],
        "train_loss": [], "train_iou": [], "train_dice": [], "train_precision": [], "train_recall": [],
        "val_loss": [],   "val_iou": [],   "val_dice": [],   "val_precision": [],   "val_recall": [],
        "lr": []
    }

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics   = validate_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"Learning Rate: {cur_lr:.2e}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(cur_lr)

        pd.DataFrame(history).to_csv(log_csv, index=False)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), save_dir / "UNET_best_model.pth")
            print(f"best model IoU: {best_val_iou:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), save_dir / f"UNET_epoch_{epoch+1}.pth")

    print(f"\n Metrics log saved to: {log_csv}")
    return history




# %%
import torch
import numpy as np
from typing import Optional

def _to_tensor(x, like: Optional[torch.Tensor] = None, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be numpy.ndarray or torch.Tensor")
    x = x.to(dtype=dtype)
    if like is not None:
        x = x.to(device=like.device)
    return x

def _prep_binary(mask: torch.Tensor, threshold: float = 0.5):
    # (B,1,H,W) -> (B,H,W)
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() == 4 and mask.size(1) > 1:
        mask = mask.argmax(dim=1)

    if mask.dtype.is_floating_point:
        return mask > threshold
    else:
        return mask > 0

def dice_coeff(y_true, y_pred, eps=1e-7, threshold=0.5):
    y_true = _to_tensor(y_true, dtype=torch.float32)
    y_pred = _to_tensor(y_pred, like=y_true, dtype=torch.float32)

    gt = _prep_binary(y_true, threshold)
    pr = _prep_binary(y_pred, threshold)

    B = gt.size(0)
    gt = gt.reshape(B, -1)
    pr = pr.reshape(B, -1)

    inter = (gt & pr).sum(dim=1).float()
    denom = gt.sum(dim=1).float() + pr.sum(dim=1).float()

    both_empty = (denom == 0)
    dice = (2 * inter + eps) / (denom + eps)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)

    return dice.mean()

def iou_coeff(y_true, y_pred, eps=1e-7, threshold=0.5):
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

def precision_recall_coeff(y_true, y_pred, eps=1e-7, threshold=0.5):
    y_true = _to_tensor(y_true, dtype=torch.float32)
    y_pred = _to_tensor(y_pred, like=y_true, dtype=torch.float32)

    gt = _prep_binary(y_true, threshold)
    pr = _prep_binary(y_pred, threshold)

    B = gt.size(0)
    gt = gt.reshape(B, -1)
    pr = pr.reshape(B, -1)

    tp = (gt & pr).sum(dim=1).float()
    fp = ((~gt) & pr).sum(dim=1).float()
    fn = (gt & (~pr)).sum(dim=1).float()

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)

    both_empty = ((gt.sum(dim=1) == 0) & (pr.sum(dim=1) == 0))
    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall    = torch.where(both_empty, torch.ones_like(recall), recall)

    return precision.mean(), recall.mean()


# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_df = pd.read_csv("./data/data_path_result.csv").reset_index(drop=True)

dataset_train = CustomDataset(dataset_df)

set_seed(42)
datasets = train_val_dataset_fixed(dataset_train, seed=42)

split_dir = Path("./outputs/splits_seed42")
split_dir.mkdir(parents=True, exist_ok=True)
json.dump(
    {"train_idx": datasets["train_idx"], "val_idx": datasets["val_idx"], "test_idx": datasets["test_idx"]},
    open(split_dir / "split_idx.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2
)

loader_train = DataLoader(datasets["train"], batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
loader_val   = DataLoader(datasets["val"],   batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
loader_test  = DataLoader(datasets["test"],  batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# %%
model = UNet(in_channels=3, num_classes=2, base_ch=64).to(DEVICE)

criterion = nn.CrossEntropyLoss()
weight = torch.tensor([1.0, 10.0]).to(DEVICE)  
criterion = nn.CrossEntropyLoss(weight=weight)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)


with torch.no_grad():
    dummy_input = torch.randn(2, 3, 512, 512).to(DEVICE)
    out = model(dummy_input)
    print("U-Net output shape:", out.shape)  

history = train_model(
    model=model,
    train_loader=loader_train,
    val_loader=loader_val,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=200,                 
    save_path="./weights",
    start_epoch=0                  
)

# %%
import torch
from pathlib import Path

model = UNet(in_channels=3, num_classes=2, base_ch=64).to(DEVICE)

model_path = Path("./weights/UNET_best_model.pth")

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint)

model.eval()

print("UNet best model loaded:", model_path)

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from skimage.morphology import binary_erosion, square
import hashlib, re
import matplotlib.patches as mpatches

out_dir = Path("./outputs/unet_test_vis")
raw_dir  = out_dir / "raw_gt"
pred_dir = out_dir / "pred_overlay"

raw_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

dice_list, iou_list = [], []
precision_list, recall_list = [], []
ids_list, safe_ids = [], []
selem = square(5)


def safe_basename(x: str) -> str:
    s = str(x).replace("\\", "/")
    p = Path(s)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)
    h = hashlib.md5(s.encode()).hexdigest()[:6]
    return f"{stem}_{h}"


with torch.no_grad():

    for image_ids, imgs, masks in loader_test:

        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        preds = torch.softmax(preds, dim=1)

        a = preds.cpu().numpy()
        fg_ch = 1 if a.shape[1] > 1 else 0

        # ----- 원본 이미지 -----
        img = imgs[0].cpu().numpy().transpose(1,2,0)

        if img.max() > 1:
            img = img / 255.0

        # ----- GT -----
        m = masks[0].cpu()

        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)

        mask_t = (m > 0).to(torch.uint8)
        mask_np = mask_t.numpy()

        # ----- Prediction -----
        pred_prob = a[0, fg_ch]

        pred_bin = pred_prob >= 0.5

        eroded = binary_erosion(pred_bin, footprint=selem)

        pred = eroded.astype(np.uint8)

        # ----- Overlay 색상 -----
        tp = (mask_np == 1) & (pred == 1)
        fp = (mask_np == 0) & (pred == 1)
        fn = (mask_np == 1) & (pred == 0)

        tp_count = tp.sum()
        fp_count = fp.sum()
        fn_count = fn.sum()

        precision = tp_count / (tp_count + fp_count + 1e-8)
        recall    = tp_count / (tp_count + fn_count + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)

        overlay = np.zeros((*mask_np.shape,3), dtype=np.float32)

        overlay[tp] = [1,1,0]
        overlay[fp] = [1,0,0]
        overlay[fn] = [0,1,0]

        # ----- RAW + GT -----
        gt_overlay = np.zeros_like(overlay)
        gt_overlay[mask_np==1] = [0,1,0]

        left_viz  = np.clip(0.5*img + 0.5*gt_overlay,0,1)
        right_viz = np.clip(0.5*img + 0.5*overlay,0,1)

        # ----- metric -----
        pred_t = torch.from_numpy(pred).to(mask_t.device).type_as(mask_t)

        dc = dice_coeff(mask_t.unsqueeze(0), pred_t.unsqueeze(0)).item()
        io = iou_coeff(mask_t.unsqueeze(0), pred_t.unsqueeze(0)).item()

        orig_id  = str(image_ids[0])
        title_id = Path(orig_id.replace("\\","/")).name
        file_id  = safe_basename(orig_id)

        dice_list.append(dc)
        iou_list.append(io)
        ids_list.append(orig_id)
        safe_ids.append(file_id)

        # ----- RAW+GT 저장 -----
        fig1, ax1 = plt.subplots(figsize=(5,5))

        ax1.imshow(left_viz)
        ax1.set_title(f"{title_id} (RAW+GT)")
        ax1.axis("off")

        fig1.savefig(raw_dir / f"{file_id}.png", dpi=150)
        plt.close(fig1)

        # ----- Pred overlay 저장 -----
        fig2, ax2 = plt.subplots(figsize=(5,5))

        ax2.imshow(right_viz)
        ax2.set_title(f"{title_id} (GT/Pred/TP)")
        ax2.axis("off")

        tp_patch = mpatches.Patch(color='yellow', label='TP')
        fp_patch = mpatches.Patch(color='red', label='FP')
        fn_patch = mpatches.Patch(color='green', label='FN')

        ax2.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower right')

        fig2.savefig(pred_dir / f"{file_id}.png", dpi=150)

        plt.close(fig2)


# ----- 평균 metric -----
mean_dice = float(np.mean(dice_list))
mean_iou  = float(np.mean(iou_list))

print(f"[전체 평균] Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")


# ----- CSV 저장 -----
df = pd.DataFrame({
    "image_id": ids_list,
    "safe_id":  safe_ids,
    "dice": dice_list,
    "iou": iou_list,
    "precision": precision_list,
    "recall": recall_list
})

df.loc[len(df)] = [
    "mean",
    "mean",
    mean_dice,
    mean_iou,
    np.mean(precision_list),
    np.mean(recall_list)
]

df.to_csv(out_dir / "metrics_test.csv", index=False)

# %%
precision_list, recall_list 
