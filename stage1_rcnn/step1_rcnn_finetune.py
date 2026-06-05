"""
Experiment A — Step 1: Fine-tune Faster R-CNN on BTCV (Pancreas Detection)
===========================================================================
Trains a COCO-pretrained Faster R-CNN to detect the pancreas on full-size
(512×512) BTCV CT slices. Bounding box labels are derived from the GT masks.

WHY:
  The paper's BTCV evaluation used ground-truth boxes to crop the ROI before
  feeding into TransUNet. Reviewer 1 (Round 2) flagged this as non-automatic.
  This script trains a detector so the pipeline becomes fully end-to-end:
    Full BTCV slice → Faster R-CNN → predicted box → crop → TransUNet → mask

TRAINING DATA:
  - 24 BTCV training cases (img0001-0010, img0021-0034)
  - Only pancreas-containing slices (pancreas area >= 0.5% of 512×512 image)
  - GT boxes derived slice-by-slice from segmentation masks (label == 11)

MODEL:
  - fasterrcnn_resnet50_fpn (torchvision), pretrained on COCO
  - Box predictor head replaced for 2 classes: background + pancreas
  - Backbone frozen for first 5 epochs, then fully fine-tuned

VALIDATION:
  - 6 BTCV val cases (img0035-0040), same split as dataset_0.json
  - Metric: mean IoU between top-1 predicted box and GT box

Run:
    conda activate medseg
    python experiment_a/step1_rcnn_finetune.py

Output:
    experiment_a/rcnn_model/rcnn_best.pth    <- best val IoU checkpoint
    experiment_a/rcnn_train_log.csv          <- epoch-by-epoch log
"""

import sys
import random
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[1]   # repo root
BTCV_DIR = BASE / "data" / "btcv_datasets"
IMG_DIR  = BTCV_DIR / "imagesTr"
LBL_DIR  = BTCV_DIR / "labelsTr"
EXP_DIR  = BASE / "outputs"
OUT_DIR  = EXP_DIR / "rcnn_model"
CKPT_OUT = OUT_DIR / "rcnn_best.pth"
LOG_CSV  = EXP_DIR / "rcnn_train_log.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
HU_MIN         = -125
HU_MAX         =  225
IMG_SIZE       =  512
AREA_THR       =  0.005   # pancreas must be >= 0.5% of slice area to include
PANCREAS_LABEL =  11
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS   = 30
BATCH_SIZE   = 4
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
FREEZE_EPOCHS = 5   # freeze backbone for first N epochs

TRAIN_CASES = {
    "img0001","img0002","img0003","img0004","img0005",
    "img0006","img0007","img0008","img0009","img0010",
    "img0021","img0022","img0023","img0024","img0025",
    "img0026","img0027","img0028","img0029","img0030",
    "img0031","img0032","img0033","img0034",
}
VAL_CASES = {"img0035","img0036","img0037","img0038","img0039","img0040"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def hu_window(vol):
    arr = np.clip(vol, HU_MIN, HU_MAX)
    return ((arr - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)   # [0, 1]


def get_tight_box(mask_2d):
    """Return (x1, y1, x2, y2) tight bounding box from a binary mask slice.
    Returns None if no foreground pixels."""
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    # ensure minimum box size of 4px
    if x2 - x1 < 4: x2 = min(IMG_SIZE - 1, x1 + 4)
    if y2 - y1 < 4: y2 = min(IMG_SIZE - 1, y1 + 4)
    return (float(x1), float(y1), float(x2), float(y2))


def load_case_slices(case_id, is_train=True):
    """Load all pancreas-containing slices for one case.
    Returns list of (image_tensor, target_dict) tuples."""
    img_path = IMG_DIR / f"{case_id}.nii"
    lbl_path = LBL_DIR / f"label{case_id[3:]}.nii"   # img0001 → label0001.nii

    if not img_path.exists() or not lbl_path.exists():
        print(f"[WARN] Missing files for {case_id}, skipping.")
        return []

    img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    lbl_vol = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)
    pan_vol = (lbl_vol == PANCREAS_LABEL).astype(np.uint8)

    items = []
    for z in range(img_vol.shape[2]):
        mask_s = pan_vol[:, :, z]
        area_frac = mask_s.sum() / (IMG_SIZE * IMG_SIZE)
        if area_frac < AREA_THR:
            continue

        box = get_tight_box(mask_s)
        if box is None:
            continue

        img_s = hu_window(img_vol[:, :, z])                      # (512,512) [0,1]
        img_t = torch.from_numpy(
            np.stack([img_s, img_s, img_s], axis=0)              # (3,512,512)
        )

        target = {
            "boxes":  torch.tensor([box],  dtype=torch.float32),  # (1,4)
            "labels": torch.tensor([1],    dtype=torch.int64),     # 1 = pancreas
        }
        items.append((img_t, target, case_id, z))

    return items


# ── Dataset ────────────────────────────────────────────────────────────────────
class BTCVDetectionDataset(Dataset):
    def __init__(self, case_ids, augment=False):
        self.items   = []
        self.augment = augment
        for cid in sorted(case_ids):
            self.items.extend(load_case_slices(cid))
        print(f"  Loaded {len(self.items)} slices from {len(case_ids)} cases.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_t, target, case_id, z = self.items[idx]

        if self.augment and random.random() > 0.5:
            # Horizontal flip — must mirror the box x-coordinates too
            img_t = torch.flip(img_t, dims=[2])
            boxes = target["boxes"].clone()
            x1, y1, x2, y2 = boxes[0]
            boxes[0] = torch.tensor(
                [IMG_SIZE - 1 - x2, y1, IMG_SIZE - 1 - x1, y2]
            )
            target = {"boxes": boxes, "labels": target["labels"]}

        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


def freeze_backbone(model, freeze=True):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = not freeze


# ── Validation — mean IoU of top-1 box vs GT box ──────────────────────────────
def box_iou_single(pred_box, gt_box):
    """IoU between two boxes, each (x1,y1,x2,y2)."""
    ix1 = max(pred_box[0], gt_box[0])
    iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2])
    iy2 = min(pred_box[3], gt_box[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
    area_g = (gt_box[2]-gt_box[0])     * (gt_box[3]-gt_box[1])
    union  = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def validate(model, case_ids):
    model.eval()
    iou_list = []
    for cid in sorted(case_ids):
        slices = load_case_slices(cid, is_train=False)
        for img_t, target, _, _ in slices:
            img_t = img_t.to(DEVICE)
            preds = model([img_t])[0]
            gt_box = target["boxes"][0].tolist()

            if len(preds["boxes"]) == 0:
                iou_list.append(0.0)
                continue

            # pick highest-confidence prediction
            best_idx = preds["scores"].argmax().item()
            pred_box = preds["boxes"][best_idx].cpu().tolist()
            iou_list.append(box_iou_single(pred_box, gt_box))

    return float(np.mean(iou_list)) if iou_list else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device       : {DEVICE}")
    print(f"Train cases  : {len(TRAIN_CASES)}")
    print(f"Val cases    : {len(VAL_CASES)}")
    print(f"Epochs       : {NUM_EPOCHS}  (patience={PATIENCE})")
    print(f"Backbone frozen first {FREEZE_EPOCHS} epochs\n")

    print("Loading train dataset...")
    train_ds = BTCVDetectionDataset(TRAIN_CASES, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    print("\nBuilding Faster R-CNN (COCO pretrained)...")
    model = build_model().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    best_iou    = 0.0
    patience_ctr = 0
    log_rows    = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Val IoU':>9}  {'Best IoU':>9}")
    print("-" * 50)

    for epoch in range(1, NUM_EPOCHS + 1):

        # Freeze/unfreeze backbone
        freeze_backbone(model, freeze=(epoch <= FREEZE_EPOCHS))

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        val_iou = validate(model, VAL_CASES)
        scheduler.step()

        marker = ""
        if val_iou > best_iou:
            best_iou     = val_iou
            patience_ctr = 0
            torch.save(model.state_dict(), str(CKPT_OUT))
            marker = " <- BEST"
        else:
            patience_ctr += 1

        print(f"{epoch:>5}  {avg_loss:>11.4f}  {val_iou*100:>8.2f}%  {best_iou*100:>8.2f}%{marker}")

        log_rows.append({
            "epoch":    epoch,
            "train_loss": round(avg_loss, 4),
            "val_iou":    round(val_iou * 100, 2),
            "best_iou":   round(best_iou * 100, 2),
        })

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping: no improvement for {PATIENCE} epochs.")
            break

    pd.DataFrame(log_rows).to_csv(str(LOG_CSV), index=False)

    print(f"\n{'='*55}")
    print(f"  Faster R-CNN Training Complete")
    print(f"{'='*55}")
    print(f"  Best Val IoU : {best_iou*100:.2f}%")
    print(f"  Checkpoint   : {CKPT_OUT}")
    print(f"  Log          : {LOG_CSV}")
    print(f"{'='*55}")
    print("\nNext: python experiment_a/step2_rcnn_predict.py")
