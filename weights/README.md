# Weights

Place trained checkpoints here. The scripts expect:

- `DICE_HD_best_model_100after.pth` — NIH-pretrained TransUNet (DHD).
  Required by `evaluate_btcv/btcv_evaluate_v9.py`, `evaluate_btcv/btcv_5fold_cv.py`,
  and `inference_time.py`.

Checkpoints are git-ignored (too large for the repo). Obtain them from the
authors or the release link in the main README.
