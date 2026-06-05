"""
Experiment A — Step 4g: Standard 5-Fold Cross-Validation (All 30 Cases)
========================================================================
Standard 5-fold CV on all 30 BTCV cases:
  - 30 cases divided into 5 folds of 6
  - Each fold: 24 remaining cases  →  20 train + 4 val + 6 test
  - The 4 val cases come FROM the 24 (not extra or separate)
  - Val used only for checkpoint selection — never reported
  - 6 test cases completely withheld, evaluated once
  - Every case tested exactly once across all 5 folds
  - Fully automatic pipeline (R-CNN predicted boxes, no ground-truth)

Run:
    python experiment_a/step4g_standard_kfold.py

Output:
    experiment_a/standard_kfold/fold{k}_model.pth
    experiment_a/standard_kfold/fold{k}_log.csv
    experiment_a/standard_kfold_results.txt
    experiment_a/standard_kfold_per_case.csv
"""

import sys, random, time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
from pathlib import Path
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference_time import create_model

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE     = Path(__file__).resolve().parents[1]   # repo root
BTCV_DIR = BASE / "data" / "btcv_datasets"
IMG_DIR  = BTCV_DIR / "imagesTr"
LBL_DIR  = BTCV_DIR / "labelsTr"
EXP_DIR  = BASE / "outputs"
BOX_CSV  = EXP_DIR / "rcnn_predicted_boxes.csv"
NIH_CKPT = BASE / "weights" / "DICE_HD_best_model_100after.pth"
OUT_DIR  = EXP_DIR / "standard_kfold"
OUT_TXT  = EXP_DIR / "standard_kfold_results.txt"
OUT_CSV  = EXP_DIR / "standard_kfold_per_case.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_LR   = 1e-5
DECODER_LR   = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 4
NUM_EPOCHS   = 60
PATIENCE     = 12
VAL_THR      = 0.3    # threshold for checkpoint selection
THRESHOLD    = 0.05   # threshold for final test evaluation
IMG_SIZE     = 512
HU_MIN       = -125
HU_MAX       =  225
PANCREAS_LABEL = 11
N_FOLDS      = 5
N_VAL        = 4      # val cases taken from the 24 non-test cases
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CASES = [
    "img0001","img0002","img0003","img0004","img0005",
    "img0006","img0007","img0008","img0009","img0010",
    "img0021","img0022","img0023","img0024","img0025",
    "img0026","img0027","img0028","img0029","img0030",
    "img0031","img0032","img0033","img0034","img0035",
    "img0036","img0037","img0038","img0039","img0040",
]  # 30 cases


def make_folds(cases, n_folds, n_val, seed):
    """
    Split all cases into n_folds.
    Each fold: test = 1 group of 6
               remaining 24 = first n_val as val, rest as train
    """
    shuffled = cases.copy()
    random.Random(seed).shuffle(shuffled)
    fold_size = len(shuffled) // n_folds
    folds = []
    for k in range(n_folds):
        test      = shuffled[k * fold_size: (k + 1) * fold_size]
        remaining = [c for c in shuffled if c not in test]  # 24 cases
        val       = remaining[:n_val]                        # first 4 of the 24
        train     = remaining[n_val:]                        # remaining 20
        folds.append({"train": train, "val": val, "test": test})
    return folds


def hu_window(arr):
    arr = np.clip(arr, HU_MIN, HU_MAX)
    return ((arr - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)


def crop_and_resize(img_2d, mask_2d, x0, y0, x1, y1, size=IMG_SIZE):
    ic = img_2d[y0:y1, x0:x1]; mc = mask_2d[y0:y1, x0:x1]
    if ic.shape[0] < 2 or ic.shape[1] < 2:
        ic, mc = img_2d, mask_2d
    ir = np.array(Image.fromarray(ic).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    mr = np.array(Image.fromarray(mc).resize((size, size), Image.NEAREST), dtype=np.uint8)
    return ir, mr


def build_samples(box_df, case_list):
    samples = []
    for case_id in sorted(case_list):
        img_path = IMG_DIR / f"{case_id}.nii"
        lbl_path = LBL_DIR / f"label{case_id[3:]}.nii"
        if not img_path.exists() or not lbl_path.exists():
            continue
        img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lbl_vol = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)
        pan_vol = (lbl_vol == PANCREAS_LABEL).astype(np.uint8)
        rows    = box_df[box_df["case_id"] == case_id]
        for _, row in rows.iterrows():
            z  = int(row["slice_z"])
            x0, y0, x1, y1 = int(row["crop_x0"]), int(row["crop_y0"]), int(row["crop_x1"]), int(row["crop_y1"])
            ic, mc = crop_and_resize(hu_window(img_vol[:, :, z]), pan_vol[:, :, z], x0, y0, x1, y1)
            samples.append((np.stack([ic, ic, ic], axis=2), (mc > 0).astype(np.uint8)))
    return samples


class PredBoxDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples; self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_rgb, mask_bin = self.samples[idx]
        img_pil  = Image.fromarray(img_rgb, mode="RGB")
        mask_pil = Image.fromarray(mask_bin * 255, mode="L")
        if self.augment:
            if random.random() > 0.5:
                img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img_pil  = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.3:
                ang = random.uniform(-15, 15)
                img_pil  = img_pil.rotate(ang,  resample=Image.BILINEAR, fillcolor=0)
                mask_pil = mask_pil.rotate(ang, resample=Image.NEAREST,  fillcolor=0)
            if random.random() > 0.5:
                img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.90, 1.10))
        img_arr  = np.array(img_pil,  dtype=np.float32) / 255.0
        mask_arr = np.array(mask_pil, dtype=np.uint8)
        return (torch.from_numpy(np.transpose(img_arr, (2, 0, 1))),
                torch.from_numpy((mask_arr > 127).astype(np.int64)))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        tgt   = targets.float()
        inter = (probs * tgt).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


def compute_val_dsc(model, loader, device, thr=VAL_THR):
    model.eval(); dscs = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1, :, :]
            preds = (probs > thr).float()
            for pred, gt in zip(preds, masks.float()):
                tp = (pred * gt).sum().item()
                fp = (pred * (1 - gt)).sum().item()
                fn = ((1 - pred) * gt).sum().item()
                dscs.append((2 * tp) / (2 * tp + fp + fn + 1e-7))
    return float(np.mean(dscs))


def evaluate_test(model, box_df, case_list, device, thr=THRESHOLD):
    model.eval(); records = []
    for case_id in sorted(case_list):
        img_path = IMG_DIR / f"{case_id}.nii"
        lbl_path = LBL_DIR / f"label{case_id[3:]}.nii"
        if not img_path.exists() or not lbl_path.exists():
            continue
        img_vol = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        lbl_vol = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.int32)
        pan_vol = (lbl_vol == PANCREAS_LABEL).astype(np.uint8)
        rows    = box_df[box_df["case_id"] == case_id]
        # Accumulate TP/FP/FN across all slices for case-level Precision & Recall
        case_tp = case_fp = case_fn = 0
        dscs = []
        for _, row in rows.iterrows():
            z  = int(row["slice_z"])
            x0, y0, x1, y1 = int(row["crop_x0"]), int(row["crop_y0"]), int(row["crop_x1"]), int(row["crop_y1"])
            ic, mc = crop_and_resize(hu_window(img_vol[:, :, z]), pan_vol[:, :, z], x0, y0, x1, y1)
            img_t  = torch.from_numpy(
                np.transpose(np.stack([ic, ic, ic], axis=2).astype(np.float32) / 255.0, (2, 0, 1))
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(img_t), dim=1)[0, 1].cpu().numpy()
            pred = (probs > thr).astype(np.uint8)
            gt   = (mc > 0).astype(np.uint8)
            tp = int((pred * gt).sum())
            fp = int((pred * (1 - gt)).sum())
            fn = int(((1 - pred) * gt).sum())
            case_tp += tp; case_fp += fp; case_fn += fn
            dscs.append((2 * tp) / (2 * tp + fp + fn + 1e-7))
        case_dsc  = float(np.mean(dscs)) * 100 if dscs else 0.0
        case_prec = case_tp / (case_tp + case_fp + 1e-7) * 100
        case_rec  = case_tp / (case_tp + case_fn + 1e-7) * 100
        records.append({
            "case_id":   case_id,
            "mean_dsc":  round(case_dsc,  2),
            "precision": round(case_prec, 2),
            "recall":    round(case_rec,  2),
            "n_slices":  len(dscs),
        })
    return records


def get_param_groups(model):
    m = model.module if hasattr(model, "module") else model
    enc, dec = [], []
    for name, param in m.named_parameters():
        (enc if any(k in name for k in ["transformer", "root", "body"]) else dec).append(param)
    return [{"params": enc, "lr": ENCODER_LR}, {"params": dec, "lr": DECODER_LR}]


def train_fold(fold_idx, train_cases, val_cases, test_cases, box_df):
    print(f"\n{'='*62}")
    print(f"  FOLD {fold_idx + 1}/5")
    print(f"  Train : {len(train_cases)} cases — {sorted(train_cases)}")
    print(f"  Val   : {len(val_cases)} cases  — {sorted(val_cases)}  (checkpoint selection)")
    print(f"  Test  : {len(test_cases)} cases  — {sorted(test_cases)}")
    print(f"{'='*62}")

    print("  Loading crops...")
    train_samples = build_samples(box_df, train_cases)
    val_samples   = build_samples(box_df, val_cases)
    print(f"  Train slices: {len(train_samples)}  |  Val slices: {len(val_samples)}")

    train_loader = DataLoader(
        PredBoxDataset(train_samples, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        PredBoxDataset(val_samples, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = create_model()
    ckpt  = torch.load(str(NIH_CKPT), map_location=DEVICE)
    if next(iter(ckpt)).startswith("module."):
        ckpt = OrderedDict((k.replace("module.", "", 1), v) for k, v in ckpt.items())
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)

    optimizer    = torch.optim.AdamW(get_param_groups(model), weight_decay=WEIGHT_DECAY)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion    = DiceLoss().to(DEVICE)
    ckpt_path    = OUT_DIR / f"fold{fold_idx + 1}_model.pth"
    log_rows     = []
    best_val     = -1.0
    patience_cnt = 0
    best_epoch   = 0

    print(f"\n  {'Epoch':>5}  {'Loss':>8}  {'Val DSC':>9}  {'Best':>6}  {'ETA':>10}")
    print(f"  {'-'*52}")

    epoch_times = []
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train(); total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        v_dsc    = compute_val_dsc(model, val_loader, DEVICE)
        improved = v_dsc > best_val
        if improved:
            best_val = v_dsc; best_epoch = epoch; patience_cnt = 0
            torch.save(model.state_dict(), str(ckpt_path))
        else:
            patience_cnt += 1

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        eta_sec = np.mean(epoch_times) * min(NUM_EPOCHS - epoch, PATIENCE - patience_cnt + 1)
        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
        marker  = " *" if improved else ""

        print(f"  {epoch:>5}  {avg_loss:>8.4f}  {v_dsc*100:>8.2f}%{marker:<2}  "
              f"{best_val*100:>5.2f}%  {eta_str:>10}")
        log_rows.append({"epoch": epoch, "train_loss": round(avg_loss, 4),
                         "val_dsc": round(v_dsc * 100, 2)})

        if patience_cnt >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}. Best epoch: {best_epoch}")
            break

    pd.DataFrame(log_rows).to_csv(str(OUT_DIR / f"fold{fold_idx + 1}_log.csv"), index=False)
    print(f"  Best val DSC : {best_val * 100:.2f}%  at epoch {best_epoch}")

    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    print(f"\n  Evaluating {len(test_cases)} test cases...")
    records   = evaluate_test(model, box_df, test_cases, DEVICE)
    fold_dsc  = float(np.mean([r["mean_dsc"]  for r in records]))
    fold_prec = float(np.mean([r["precision"] for r in records]))
    fold_rec  = float(np.mean([r["recall"]    for r in records]))
    for r in records:
        print(f"    {r['case_id']}  DSC={r['mean_dsc']:.2f}%  "
              f"Prec={r['precision']:.2f}%  Rec={r['recall']:.2f}%  ({r['n_slices']} slices)")
    print(f"  Fold {fold_idx + 1} → DSC={fold_dsc:.2f}%  Prec={fold_prec:.2f}%  Rec={fold_rec:.2f}%")
    return records, best_val * 100, best_epoch


if __name__ == "__main__":
    print(f"Device     : {DEVICE}")
    print(f"Total cases: {len(ALL_CASES)}")
    print(f"Per fold   : 20 train  |  4 val (checkpoint selection)  |  6 test")
    print(f"Epochs     : up to {NUM_EPOCHS}  (early stop patience={PATIENCE})")

    box_df = pd.read_csv(str(BOX_CSV))
    folds  = make_folds(ALL_CASES, N_FOLDS, N_VAL, SEED)

    print(f"\nFold assignments (seed={SEED}):")
    for k, f in enumerate(folds):
        print(f"  Fold {k+1}  train={len(f['train'])}  val={sorted(f['val'])}  test={sorted(f['test'])}")

    all_records    = []
    fold_summaries = []

    for k, fold in enumerate(folds):
        records, best_val_dsc, best_ep = train_fold(
            k, fold["train"], fold["val"], fold["test"], box_df
        )
        all_records.extend(records)
        fold_summaries.append({
            "fold":          k + 1,
            "best_val_dsc":  round(best_val_dsc, 2),
            "best_epoch":    best_ep,
            "test_dsc_mean": round(float(np.mean([r["mean_dsc"]  for r in records])), 2),
            "test_prec_mean":round(float(np.mean([r["precision"] for r in records])), 2),
            "test_rec_mean": round(float(np.mean([r["recall"]    for r in records])), 2),
        })

    all_dscs  = [r["mean_dsc"]  for r in all_records]
    all_precs = [r["precision"] for r in all_records]
    all_recs  = [r["recall"]    for r in all_records]
    overall_mean = float(np.mean(all_dscs))
    overall_std  = float(np.std(all_dscs))

    overall_prec_mean = float(np.mean(all_precs))
    overall_prec_std  = float(np.std(all_precs))
    overall_rec_mean  = float(np.mean(all_recs))
    overall_rec_std   = float(np.std(all_recs))

    summary = f"""
{'='*72}
  Experiment A — Standard 5-Fold Cross-Validation
  30 BTCV cases  |  5 folds of 6 test cases
  Per fold: 20 train  +  4 val (checkpoint selection)  +  6 test
  All 30 cases tested exactly once
  Fully automatic: R-CNN predicted boxes, no ground-truth input
{'='*72}

  Fold breakdown:
    {'Fold':<6}  {'BestVal':>8}  {'BestEp':>7}  {'DSC':>8}  {'Precision':>11}  {'Recall':>8}
    {'-'*58}
"""
    for fs in fold_summaries:
        summary += (f"    Fold {fs['fold']}  {fs['best_val_dsc']:>7.2f}%"
                    f"  ep {fs['best_epoch']:>2}  "
                    f"  {fs['test_dsc_mean']:>7.2f}%"
                    f"  {fs['test_prec_mean']:>10.2f}%"
                    f"  {fs['test_rec_mean']:>7.2f}%\n")

    summary += f"""
  --- OVERALL RESULT (all 30 cases, each tested once) ---
  DSC       : {overall_mean:.2f}% +/- {overall_std:.2f}%
  Precision : {overall_prec_mean:.2f}% +/- {overall_prec_std:.2f}%
  Recall    : {overall_rec_mean:.2f}% +/- {overall_rec_std:.2f}%
  Min DSC   : {min(all_dscs):.2f}%   Max DSC: {max(all_dscs):.2f}%

  --- Per-case breakdown ---
  {'Case':<12}  {'DSC':>8}  {'Precision':>11}  {'Recall':>8}  {'Slices':>7}
  {'-'*54}
"""
    for r in sorted(all_records, key=lambda x: x["case_id"]):
        summary += (f"  {r['case_id']:<12}  {r['mean_dsc']:>7.2f}%"
                    f"  {r['precision']:>10.2f}%"
                    f"  {r['recall']:>7.2f}%"
                    f"  {r['n_slices']:>6}\n")

    summary += f"\n{'='*64}\n"

    print("\n" + summary)
    OUT_TXT.write_text(summary)
    pd.DataFrame(all_records)[["case_id","mean_dsc","precision","recall","n_slices"]].to_csv(
        str(OUT_CSV), index=False
    )
    print(f"Saved:\n  {OUT_TXT}\n  {OUT_CSV}")
