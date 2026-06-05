"""
Experiment A — Step 2: Run Fine-Tuned Faster R-CNN on All 30 BTCV Cases
=========================================================================
Loads the fine-tuned Faster R-CNN from Step 1 and runs it on every
pancreas-containing slice of all 30 labeled BTCV cases.

For each slice the detector outputs a predicted bounding box. That box is
then expanded by 2.0× (same scale as the GT-box protocol in v3) and saved
to a CSV. Step 3 reads this CSV to crop slices without ever touching the GT mask.

FALLBACK RULE:
  If the detector finds no box on a slice (score below threshold), the fallback
  is the full 512×512 image (equivalent to no crop). This is logged so we can
  audit how often the detector fails.

Run:
    conda activate medseg
    python experiment_a/step2_rcnn_predict.py

Output:
    experiment_a/rcnn_predicted_boxes.csv   <- one row per pancreas slice
"""

import sys
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[1]   # repo root
BTCV_DIR = BASE / "data" / "btcv_datasets"
IMG_DIR  = BTCV_DIR / "imagesTr"
LBL_DIR  = BTCV_DIR / "labelsTr"
EXP_DIR  = BASE / "outputs"
CKPT     = EXP_DIR / "rcnn_model" / "rcnn_best.pth"
OUT_CSV  = EXP_DIR / "rcnn_predicted_boxes.csv"

# ── Settings ───────────────────────────────────────────────────────────────────
HU_MIN         = -125
HU_MAX         =  225
IMG_SIZE       =  512
AREA_THR       =  0.005
PANCREAS_LABEL =  11
BOX_SCALE      =  2.0
SCORE_THR      =  0.3    # minimum detection confidence; below this → fallback
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CASES = [
    "img0001","img0002","img0003","img0004","img0005",
    "img0006","img0007","img0008","img0009","img0010",
    "img0021","img0022","img0023","img0024","img0025",
    "img0026","img0027","img0028","img0029","img0030",
    "img0031","img0032","img0033","img0034",
    "img0035","img0036","img0037","img0038","img0039","img0040",
]

VAL_CASES = {"img0035","img0036","img0037","img0038","img0039","img0040"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def hu_window(vol):
    arr = np.clip(vol, HU_MIN, HU_MAX)
    return ((arr - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def expand_box(x1, y1, x2, y2, scale=BOX_SCALE, size=IMG_SIZE):
    """Expand a tight box by scale factor, clamped to image bounds.
    Mirrors the get_expanded_box logic from btcv_load_preprocess_v3.py."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    nx0 = int(max(0,    cx - w / 2))
    nx1 = int(min(size, cx + w / 2))
    ny0 = int(max(0,    cy - h / 2))
    ny1 = int(min(size, cy + h / 2))
    if nx1 - nx0 < 4: nx0, nx1 = 0, size
    if ny1 - ny0 < 4: ny0, ny1 = 0, size
    return nx0, ny0, nx1, ny1   # x0, y0, x1, y1 (for crop: img[y0:y1, x0:x1])


# ── Model loader ───────────────────────────────────────────────────────────────
def load_detector(ckpt_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not CKPT.exists():
        print(f"[ERROR] Checkpoint not found: {CKPT}")
        print("        Run step1_rcnn_finetune.py first.")
        sys.exit(1)

    print(f"Device    : {DEVICE}")
    print(f"Checkpoint: {CKPT}")
    print(f"Score thr : {SCORE_THR}")
    print(f"Box scale : {BOX_SCALE}x\n")

    model = load_detector(CKPT)

    records      = []
    total_slices = 0
    fallback_cnt = 0

    for case_id in tqdm(ALL_CASES, desc="Cases"):
        img_path = IMG_DIR / f"{case_id}.nii"
        lbl_path = LBL_DIR / f"label{case_id[3:]}.nii"
        split    = "val" if case_id in VAL_CASES else "train"

        if not img_path.exists() or not lbl_path.exists():
            print(f"[WARN] Missing: {case_id}")
            continue

        img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lbl_vol = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)
        pan_vol = (lbl_vol == PANCREAS_LABEL).astype(np.uint8)
        img_win = hu_window(img_vol)

        for z in range(img_vol.shape[2]):
            mask_s = pan_vol[:, :, z]
            area_frac = mask_s.sum() / (IMG_SIZE * IMG_SIZE)
            if area_frac < AREA_THR:
                continue   # skip non-pancreas slices (standard eval protocol)

            total_slices += 1

            # Build tensor for detector
            img_s = img_win[:, :, z]
            img_t = torch.from_numpy(
                np.stack([img_s, img_s, img_s], axis=0)   # (3,512,512)
            ).unsqueeze(0).to(DEVICE)                      # (1,3,512,512)

            with torch.no_grad():
                preds = model(img_t)[0]

            # Select best prediction above score threshold
            used_fallback = False
            if len(preds["boxes"]) > 0:
                scores = preds["scores"].cpu().numpy()
                boxes  = preds["boxes"].cpu().numpy()
                # filter by score threshold
                keep = scores >= SCORE_THR
                if keep.sum() > 0:
                    best = scores[keep].argmax()
                    bx1, by1, bx2, by2 = boxes[keep][best]
                    score = float(scores[keep][best])
                else:
                    bx1, by1, bx2, by2 = 0, 0, IMG_SIZE, IMG_SIZE
                    score = 0.0
                    used_fallback = True
                    fallback_cnt += 1
            else:
                bx1, by1, bx2, by2 = 0, 0, IMG_SIZE, IMG_SIZE
                score = 0.0
                used_fallback = True
                fallback_cnt += 1

            # Expand predicted box by 2.0× (same protocol as GT-box preprocessing)
            cx0, cy0, cx1, cy1 = expand_box(bx1, by1, bx2, by2)

            records.append({
                "case_id":      case_id,
                "split":        split,
                "slice_z":      z,
                "det_x1":       round(bx1, 1),
                "det_y1":       round(by1, 1),
                "det_x2":       round(bx2, 1),
                "det_y2":       round(by2, 1),
                "det_score":    round(score, 4),
                "crop_x0":      cx0,
                "crop_y0":      cy0,
                "crop_x1":      cx1,
                "crop_y1":      cy1,
                "used_fallback": used_fallback,
            })

    df = pd.DataFrame(records)
    df.to_csv(str(OUT_CSV), index=False)

    fallback_pct = fallback_cnt / max(total_slices, 1) * 100

    print(f"\n{'='*55}")
    print(f"  Faster R-CNN Prediction Complete")
    print(f"{'='*55}")
    print(f"  Total pancreas slices  : {total_slices}")
    print(f"  Fallback (no detection): {fallback_cnt}  ({fallback_pct:.1f}%)")
    print(f"  Saved to               : {OUT_CSV}")
    print(f"{'='*55}")

    # Summary per split
    for sp in ["train", "val"]:
        sub = df[df["split"] == sp]
        fb  = sub["used_fallback"].sum()
        print(f"  {sp.upper()} — {len(sub)} slices, {fb} fallbacks ({fb/len(sub)*100:.1f}%)")

    print(f"\nNext: python experiment_a/step3_automated_evaluate.py")
