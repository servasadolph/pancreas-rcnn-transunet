"""
BTCV Abdominal CT — Load & Preprocess
======================================
Converts BTCV 3D NIfTI volumes → 2D PNG slices compatible with our NIH-trained model.
Uses ALL 30 labeled cases (imagesTr + labelsTr): training + validation from dataset_0.json.

Key differences from Task07 preprocessing:
  - Files are .nii (not .nii.gz) — nibabel handles both identically
  - Pancreas is label 11 only (not merge of 1+2 like Task07)
  - Area filter: 0.5% (same as Task07 — BTCV also has wide FOV, pancreas max ~2%)
  - No bounding box crop — full 512×512 image goes to model (same as Task07 v1)

Output structure:
  btcv_dataset/
    train_img/    ← CT slice PNGs  (3-channel RGB, 512×512, uint8 0–255)
    train_mask/   ← Binary mask PNGs (grayscale: 0=background, 255=pancreas)

Then generates btcv_data_path.csv for use with btcv_evaluate.py.

Steps:
  1. Read dataset_0.json → get all 30 labeled case pairs
  2. Load 3D .nii volume (nibabel)
  3. Apply HU windowing [−125, 225]  ← same as NIH pipeline
  4. Extract label 11 (pancreas) → binary mask (0 or 1)
  5. Extract axial slices (axis=2)
  6. Apply 0.5% area filter
  7. Convert grayscale → 3-channel RGB
  8. Save PNG image + PNG mask
  9. Write btcv_data_path.csv

Run:
  conda activate medseg
  python btcv_load_preprocess.py

Expected output: ~360 slices total (~12 per volume average)
Expected time:   5–15 minutes depending on disk speed
"""

import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[1]   # repo root
BTCV_DIR  = BASE / "data" / "btcv_datasets"
IMG_DIR   = BTCV_DIR / "imagesTr"
LBL_DIR   = BTCV_DIR / "labelsTr"
OUT_DIR   = BASE / "data" / "btcv_dataset"

OUT_IMG   = OUT_DIR / "train_img"
OUT_MASK  = OUT_DIR / "train_mask"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
HU_MIN    = -125      # same HU window used in NIH pipeline — must not change
HU_MAX    =  225
IMG_SIZE  =  512      # BTCV is already 512×512 in-plane
AREA_THR  =  0.005   # 0.5% area threshold
                      # WHY NOT 4%: BTCV uses wider FOV than NIH (abdomen-focused).
                      # Pancreas only covers 0.5–2% of each 512×512 slice.
                      # Using 4% discards ALL slices (verified: 0 slices pass at 4%).
                      # 0.5% = ~1,310 pancreas pixels — enough to be meaningful.

PANCREAS_LABEL = 11   # in BTCV 13-class labels, pancreas = label 11

# ── HU windowing ───────────────────────────────────────────────────────────────
def apply_hu_window(volume_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Clip raw CT HU values to [hu_min, hu_max], normalize to [0,1], scale to uint8.
    Must be identical to NIH pipeline so the model sees consistent pixel value ranges.
    """
    arr = np.clip(volume_array, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)   # → [0.0, 1.0]
    return (arr * 255).astype(np.uint8)          # → [0, 255]

# ── Extract binary pancreas mask ───────────────────────────────────────────────
def extract_pancreas_mask(label_array):
    """
    BTCV has 13 organ labels (0–13). We only care about label 11 = pancreas.
    All other organs (liver, spleen, kidneys, etc.) become background (0).

    Unlike Task07 where we merged labels 1+2, here we simply isolate label 11.
    """
    return (label_array == PANCREAS_LABEL).astype(np.uint8)   # 0 or 1

# ── Area filter ────────────────────────────────────────────────────────────────
def pancreas_area_ok(mask_2d, threshold=AREA_THR):
    """
    Returns True if pancreas pixels occupy >= 0.5% of the 512×512 image.
    Removes slices where pancreas is a tiny edge sliver (not useful for evaluation).
    """
    total_pixels    = mask_2d.shape[0] * mask_2d.shape[1]
    pancreas_pixels = np.sum(mask_2d > 0)
    return (pancreas_pixels / total_pixels) >= threshold

# ── Load dataset_0.json ────────────────────────────────────────────────────────
json_path = BTCV_DIR / "dataset_0.json"
with open(json_path, "r") as f:
    meta = json.load(f)

# Use ALL labeled cases: training (24) + validation (6) = 30 total
# We do not retrain, so train/val split does not matter
all_labeled = meta["training"] + meta["validation"]
print(f"Labeled cases from dataset_0.json: {len(all_labeled)}")
print(f"  Training cases  : {len(meta['training'])}")
print(f"  Validation cases: {len(meta['validation'])}")
print(f"  Total for eval  : {len(all_labeled)}")

# ── Main preprocessing loop ────────────────────────────────────────────────────
records        = []
total_volumes  = 0
total_kept     = 0
total_filtered = 0
missing        = 0

print(f"\nSettings:")
print(f"  HU window   : [{HU_MIN}, {HU_MAX}]")
print(f"  Area filter : {AREA_THR*100:.1f}%")
print(f"  Pancreas label: {PANCREAS_LABEL}")
print(f"  Output dir  : {OUT_DIR}")
print()

for pair in tqdm(all_labeled, desc="Volumes"):
    # dataset_0.json paths look like "imagesTr/img0001.nii"
    img_fname = Path(pair["image"]).name      # e.g. "img0001.nii"
    lbl_fname = Path(pair["label"]).name      # e.g. "label0001.nii"

    img_path = IMG_DIR / img_fname
    lbl_path = LBL_DIR / lbl_fname

    if not img_path.exists():
        print(f"[WARN] Missing image: {img_path}")
        missing += 1
        continue
    if not lbl_path.exists():
        print(f"[WARN] Missing label: {lbl_path}")
        missing += 1
        continue

    # -- Load NIfTI volumes
    img_nib = nib.load(str(img_path))
    lbl_nib = nib.load(str(lbl_path))

    img_vol = img_nib.get_fdata(dtype=np.float32)        # (512, 512, D) — raw HU
    lbl_vol = np.asarray(lbl_nib.dataobj).astype(np.int32)  # (512, 512, D) — 0–13

    n_slices = img_vol.shape[2]
    case_id  = img_fname.replace(".nii", "")              # e.g. "img0001"

    # -- Apply HU windowing to the whole volume at once (faster than per-slice)
    img_windowed = apply_hu_window(img_vol)               # (512, 512, D) uint8

    # -- Extract pancreas mask from label volume
    lbl_binary = extract_pancreas_mask(lbl_vol)           # (512, 512, D) uint8: 0 or 1

    vol_kept = 0

    for z in range(n_slices):
        img_slice = img_windowed[:, :, z]    # (512, 512) uint8
        lbl_slice = lbl_binary[:, :, z]      # (512, 512) uint8: 0 or 1

        # -- 0.5% area filter: skip slices with barely any pancreas
        if not pancreas_area_ok(lbl_slice):
            total_filtered += 1
            continue

        # -- Resize to 512×512 if needed (BTCV is always 512×512 in-plane, but safety check)
        if img_slice.shape != (IMG_SIZE, IMG_SIZE):
            img_slice = np.array(
                Image.fromarray(img_slice).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            )
            lbl_slice = np.array(
                Image.fromarray(lbl_slice).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            )

        # -- Convert grayscale → 3-channel RGB
        # ResNet-50 backbone expects 3-channel input. CT is grayscale.
        # Duplicate the single channel 3 times — same approach as NIH preprocessing.
        img_rgb = np.stack([img_slice, img_slice, img_slice], axis=2)   # (512, 512, 3)

        # -- File names: btcv_img0001_z0045.png
        slice_name    = f"btcv_{case_id}_z{z:04d}"
        img_out_path  = OUT_IMG  / f"{slice_name}.png"
        mask_out_path = OUT_MASK / f"{slice_name}_mask.png"

        # -- Save image (RGB) and mask (grayscale 0/255)
        Image.fromarray(img_rgb,         mode="RGB").save(str(img_out_path))
        Image.fromarray(lbl_slice * 255, mode="L"  ).save(str(mask_out_path))
        # Mask saved as 0=black / 255=white
        # The model dataloader applies: mask > 8 → 1, which correctly converts 255 → 1

        records.append({
            "case_id":    case_id,
            "slice_z":    z,
            "train_img":  str(img_out_path),
            "train_mask": str(mask_out_path),
        })
        vol_kept += 1

    total_kept    += vol_kept
    total_volumes += 1

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = BASE / "btcv_data_path.csv"
pd.DataFrame(records).to_csv(str(csv_path), index=False)

# ── Final summary ──────────────────────────────────────────────────────────────
total_all_slices = sum(
    nib.load(str(IMG_DIR / Path(p["image"]).name)).header.get_data_shape()[2]
    for p in all_labeled
    if (IMG_DIR / Path(p["image"]).name).exists()
)

print(f"\n{'='*55}")
print(f"  BTCV Preprocessing Complete")
print(f"{'='*55}")
print(f"  Volumes processed    : {total_volumes} / {len(all_labeled)}")
if missing:
    print(f"  Volumes missing      : {missing}  ← check file paths")
print(f"  Total slices (raw)   : {total_all_slices}")
print(f"  Slices filtered out  : {total_filtered}  (pancreas < 0.5%)")
print(f"  Slices kept (≥0.5%)  : {total_kept}")
print(f"  Avg slices per case  : {total_kept / max(total_volumes, 1):.1f}")
print(f"  CSV saved to         : {csv_path}")
print(f"  Images saved to      : {OUT_IMG}")
print(f"  Masks  saved to      : {OUT_MASK}")
print(f"{'='*55}")
print(f"\nNext step: run btcv_evaluate.py")
