# SmartRupee: six-denomination training

This training pipeline covers ₹10, ₹20, ₹50, ₹100, ₹200 and ₹500. There is no official ₹30 banknote denomination; ₹100 is included because it was missing from the requested list.

## Important limitation

The ZIP's existing dataset is not a complete six-denomination genuine/fake dataset. RBI reference pictures are reference material, not a balanced collection of verified real-world notes, and must not be automatically treated as training examples. The current small ₹50/₹200 sample sets are not enough to claim reliable counterfeit detection. You must add verified examples for all 12 classes below. Never label an image counterfeit unless its status is independently established.

## Required folder structure

Create these folders under `dataset/all_currencies/`:

```text
dataset/all_currencies/
  train/
    genuine_10/       fake_10/
    genuine_20/       fake_20/
    genuine_50/       fake_50/
    genuine_100/      fake_100/
    genuine_200/      fake_200/
    genuine_500/      fake_500/
  validation/         # same 12 class folders
  test/               # same 12 class folders; keep untouched until final evaluation
```

Each folder contains image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). The script expects the 12 folders in each split, including the test split if you want final testing. Validation is mandatory; test is optional.

Use different physical banknotes for train, validation and test. Do not put front/back images or multiple photos of the same physical note across different splits. Keep the same camera/lighting diversity in each split. Do not create validation/test data by copying or augmenting training images.

Aim for hundreds of distinct physical examples per class as a starting data-collection target, not a guarantee of accuracy. If you cannot obtain verified counterfeit notes, do not fabricate them or scrape random images and call them fake. Mark the project as a prototype and use `inconclusive` for uncertain images.

## Install and train (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_all_currencies.py --data-dir dataset/all_currencies --epochs 25 --fine-tune --fine-tune-epochs 10
```

First run with `--weights imagenet` may download pretrained MobileNetV2 weights and needs internet. If offline, use `--weights none`; it will train from random initialization and generally needs substantially more data.

The trained model is saved under `models/all_currencies/` as `smart rupee all currencies.keras`, with class names, configuration, and optional held-out test metrics.

## Important integration note

This script creates a new 12-class model. It does not automatically replace the existing anomaly model used by the Flask application's `smartr upee/engine.py` (the actual folder is `smartr upee` without a space: `smartrupee/`). The Flask inference code must be explicitly updated to load this model and interpret its 12 class probabilities. Do not assume the existing app uses this new model until that integration is implemented and tested.

## Interpretation and safety

The model learns image patterns, not all physical security properties. A normal phone photograph cannot reliably verify every watermark, security thread, intaglio feature, or UV/IR response. Use outputs such as `genuine-like`, `possible counterfeit`, or `inconclusive`; do not claim official authentication or use a probability score as proof. Have uncertain notes checked by a bank.
