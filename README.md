# SmartRupee — Visual Banknote Screening

SmartRupee is a local Flask web application that analyzes a photo of an
Indian banknote. It attempts to align the note, identify a matching
denomination reference, compare visible regions, and report visual evidence.

> **Important:** This is an experimental visual screening aid, not an
> authentication service or a validated counterfeit detector. A result such
> as “Likely Genuine” only means that the image was consistent with the
> available references and model thresholds. It does not prove authenticity.
> A visual mismatch does not prove counterfeiting either. Check physical
> security features and consult a bank or qualified authority.

## What tools and technologies are used?

### Application and image processing

| Tool | How this project uses it |
|---|---|
| **Python** | Implements the web server, image-analysis pipeline, model training, dataset inspection, and evaluation scripts. |
| **Flask** | Serves the web page and the `/api/analyze` and `/api/status/<job_id>` endpoints. Uploaded images are analyzed in a background thread, and the browser polls for progress. |
| **HTML, CSS, JavaScript** | Build the upload/camera interface, progress display, note-region overlays, and results page. Frontend code is in `templates/` and `static/`. |
| **Bootstrap 5.3.3** | Provides layout and modal components; the template loads Bootstrap CSS and JavaScript from jsDelivr, so those assets require internet access in the browser. |
| **OpenCV (`opencv-python`)** | Reads and resizes images; denoises them; estimates note boundaries; applies perspective correction; extracts regions; detects edges and texture; calculates ORB feature matches; and creates previews and difference maps. |
| **Pillow (`Pillow`)** | Decodes uploaded JPG/PNG images and applies EXIF orientation correction before OpenCV analysis. |
| **NumPy** | Represents image pixels, feature embeddings, and model statistics as arrays and performs numerical calculations. |
| **scikit-image** | Calculates Structural Similarity (SSIM) when comparing image regions. |
| **EasyOCR** | Optionally detects visible printed text in selected regions. Text recognition does not verify a serial number or authenticate a note. If OCR cannot initialize or fails, the result reports OCR as unavailable. |

### Machine learning and evaluation

| Tool | How this project uses it |
|---|---|
| **TensorFlow / Keras** | Loads and trains the feature extractor and genuine-reference anomaly autoencoder. |
| **MobileNetV2** | Uses ImageNet-pretrained visual features to create a 1,280-dimensional representation of each 224 × 224 image. It is a feature extractor here, not a banknote-authentication model trained by RBI. |
| **Keras autoencoder** | Learns to reconstruct embeddings from genuine-reference images. Reconstruction error and distance from the reference distribution are heuristics for visual consistency. |
| **scikit-learn** | The optional experimental ₹200 model uses `StandardScaler` and `LogisticRegression` on MobileNetV2 features. Its scores are not calibrated probabilities. |
| **joblib** | Saves and loads the experimental ₹200 classifier bundle. |
| **Matplotlib** | `evaluate.py` can save training/validation metric plots when a compatible training-history file is supplied. |

The Python dependencies and minimum versions are listed in
[`requirements.txt`](./requirements.txt). Flask, NumPy, OpenCV, Pillow,
scikit-image, scikit-learn, TensorFlow, Matplotlib, and EasyOCR are Python
packages. Bootstrap is loaded by the page from a CDN and is not installed by
`pip`. The first MobileNetV2 training run may also need internet access to
download pretrained ImageNet weights if they are not already cached.

## How an image is analyzed

The main workflow is implemented in `smartrupee/engine.py`:

1. **Upload and decode:** The browser sends a JPG or PNG. The server corrects
   EXIF orientation, converts to RGB/BGR as needed, and rejects empty,
   unreadable, unsupported, or oversized uploads (maximum 12 MB).
2. **Image quality:** The engine measures image characteristics such as blur,
   brightness, glare, and resolution. A poor-quality image leads to an
   inconclusive result.
3. **Preprocess and align:** OpenCV resizes and denoises the image, detects a
   likely quadrilateral banknote boundary, and applies a perspective warp to
   a 1200 × 520 working image when a boundary is detected. If it cannot find
   the boundary, it does not claim that the note was aligned.
4. **Find a reference:** Images under `data/references/` are loaded as
   genuine-reference images. The denomination is read from the reference
   filename (for example `rbi_200_front.jpg`); `front`/`obverse` and
   `back`/`reverse` can also identify the side. ORB feature matches help pick
   a compatible denomination/side reference.
5. **Compare regions:** The engine checks note corners and selected design
   areas. OpenCV ORB descriptors are matched using a ratio test and
   RANSAC-derived homography inliers. SSIM compares the aligned regions.
   The interface reports each region as reference-consistent, visual
   mismatch, or needs review. These are image-comparison statuses, not
   security-feature verification.
6. **Compare model features:** The MobileNetV2 extractor converts the note to
   an embedding. The autoencoder and saved genuine-reference distribution
   produce a reference-consistency label, reconstruction error, and distance.
   They do not learn a validated genuine-versus-counterfeit boundary.
7. **Build the result:** Final “Likely Genuine” requires sufficient matching
   reference/region evidence and a consistent anomaly-model result.
   Otherwise the result is inconclusive. The experimental ₹200 classifier
   does not determine the final verdict. A region mismatch or an unusual model
   score must not be interpreted as proof of counterfeiting.

The browser displays progress from the backend job stages. Camera capture is
available in a supported browser on `localhost` or over HTTPS; the browser
requires permission to use the camera.

## Project layout

```text
app.py                           Flask application and upload/status API
smartrupee/engine.py             Image analysis, reference matching, and model inference
templates/index.html             Page markup
static/css/style.css             Page styles
static/js/app.js                 Upload, camera, polling, and result rendering
data/references/                 Genuine-reference photos used by the app
dataset/train/genuine/           Genuine images for shared model training
data/training_200/genuine/       User-labeled genuine examples for experimental ₹200 model
data/training_200/fake/          User-labeled fake examples for experimental ₹200 model
models/                          Saved model files and training metadata
reports/                         Generated evaluation reports and plots
train.py                         Genuine-reference anomaly model training
train_200_authenticity.py        Experimental ₹200 classifier training
prepare_dataset.py               Dataset structure and exact-duplicate audit
evaluate.py                      Held-out classifier evaluation utility
requirements.txt                 Python dependencies
```

## Install and run (Windows / VS Code)

1. Install Python 3.10–3.12 and VS Code with the Python extension.
2. Open this folder in VS Code. In PowerShell, create and activate a virtual
   environment, then install the project dependencies:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, follow your organization's policy or use
   the interpreter directly as `.\venv\Scripts\python.exe`.
3. In VS Code, select the `venv` interpreter using **Python: Select
   Interpreter**.
4. Start the server:

   ```powershell
   python app.py
   ```

5. Open <http://127.0.0.1:5000> in a browser and upload a clear JPG/PNG. Keep
   the terminal running while you use the app. Stop it with `Ctrl+C`.

The server binds to `127.0.0.1` and is intended for local use. It is not
configured as a production or public internet service.

## Reference images and dataset preparation

### Genuine references for the web app

Put readable genuine banknote photos under `data/references/`. Reference
filenames should include a denomination such as `10`, `20`, `50`, `100`,
`200`, `500`, or `2000`; using `front`/`back` in the filename helps distinguish
sides. For example:

```text
data/references/
  rbi_200_front.jpg
  rbi_200_back.png
```

Use authorized, trustworthy reference images. The app searches references
recursively. If no compatible reference is found, the result should remain
inconclusive rather than infer authenticity from denomination alone.

### Training and evaluation folders

For `train.py`, place genuine images in `dataset/train/genuine/` or supply a
different folder with `--genuine-dir`. If training a denomination-specific
model, include the denomination in filenames so the script can filter them.
For example, `200_note_01.jpg`.

`prepare_dataset.py` expects all six directories below and checks supported
image files, exact SHA-256 duplicate files, and exact duplicates spanning
splits:

```text
dataset/
  train/
    genuine/
    fake/
  validation/
    genuine/
    fake/
  test/
    genuine/
    fake/
```

Run the audit with:

```powershell
python prepare_dataset.py --dataset dataset
```

The audit checks exact byte-for-byte duplicates; it does not detect similar
photos, different crops of the same physical note, mislabeled data, or
near-duplicates. Keep all photos of a physical note within one split to avoid
leakage. Use labels verified by a qualified source and document their
provenance.

## Training the models

### Genuine-reference anomaly model

The app's default model requires the shared model files
`models/genuine_feature_extractor.keras`,
`models/genuine_reference_anomaly.keras`, and
`models/genuine_reference_distribution.npz`. Train it with at least five
readable genuine images:

```powershell
python train.py --genuine-dir dataset\train\genuine
```

Alternatively, omit `--genuine-dir` to use `dataset/train/genuine` by
default:

```powershell
python train.py
```

Training extracts MobileNetV2 features from original and augmented images,
standardizes them, and trains an autoencoder on the genuine-only embeddings.
It saves the feature extractor, anomaly model, reference distribution, and
JSON metadata under `models/`. This is a genuine-reference consistency model,
not a binary fake-note detector.

To additionally train a denomination-specific anomaly model (the app uses it
when the corresponding files exist), run for example:

```powershell
python train.py --genuine-dir data\references\user_200 --denomination 200
```

This writes `genuine_reference_anomaly_200.keras` and
`genuine_reference_distribution_200.npz` alongside the shared extractor.
Train the shared model first if the app also needs a general fallback model.
Restart the Flask server after saving or replacing model files.

### Experimental ₹200 classifier (optional)

This separate experiment needs the saved MobileNetV2 feature extractor and
user-labeled image folders:

```powershell
python train_200_authenticity.py
```

Defaults are `data/training_200/genuine/` and `data/training_200/fake/`.
Override either folder or the minimum samples per class if needed:

```powershell
python train_200_authenticity.py `
  --genuine-dir data\training_200\genuine `
  --fake-dir data\training_200\fake `
  --min-per-class 3
```

The script saves `models/currency_authenticity_200.joblib` and a JSON
training summary. Training accuracy is measured on the same images used to
fit the classifier; it is not validation accuracy. The labels must be
independently verified before any experiment, and the resulting scores are
uncalibrated. The app may display the classifier as experimental supporting
information, but it does not control the final verdict.

## Evaluation utility

`evaluate.py` evaluates a separate Keras image-classification model on a
held-out `dataset/test/` directory. It expects a model file and a JSON file
containing the model's class names. Defaults are
`models/currency_model.keras` and `models/class_names.json`:

```powershell
python evaluate.py `
  --data dataset `
  --model models\currency_model.keras `
  --class-names models\class_names.json `
  --history reports\training_history.json
```

This repository's `train.py` and `train_200_authenticity.py` do **not** create
the default `currency_model.keras` / `class_names.json` pair. Supply a
compatible classifier and its class-name file before running this evaluator.
The evaluation reports accuracy, per-class precision/recall/F1, confusion
matrix, class imbalance, and a basic overfitting assessment in
`reports/evaluation.json`. If compatible history is present, it can also
write `reports/training_validation.png`. It evaluates only the supplied
classifier and test data; it does not validate the live reference-matching
pipeline or establish real-world counterfeit-detection accuracy.

## Web API

These endpoints are implemented by `app.py`:

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the upload and analysis page. |
| `POST /api/analyze` | Accepts multipart form data with an `image` field (JPG/PNG, up to 12 MB). Returns a `job_id` or a JSON error. |
| `GET /api/status/<job_id>` | Returns progress and, when ready, the complete analysis result. Jobs are held in process memory and expire after 30 minutes. |

The app uses an in-memory job dictionary and background Python threads. A
restart clears active jobs. This simple mechanism is suitable for the local
prototype; it is not a persistent queue or multi-worker production setup.

## Model files

| File | Purpose |
|---|---|
| `models/genuine_feature_extractor.keras` | MobileNetV2-based feature extractor shared by the anomaly and experimental classifiers. |
| `models/genuine_reference_anomaly.keras` | Shared genuine-reference embedding autoencoder. |
| `models/genuine_reference_distribution.npz` | Shared embedding mean/scale, reference embeddings, and thresholds. |
| `models/genuine_reference_anomaly_<denomination>.keras` | Optional denomination-specific autoencoder, for example `_200.keras`. |
| `models/genuine_reference_distribution_<denomination>.npz` | Optional matching denomination-specific statistics and thresholds. |
| `models/currency_authenticity_200.joblib` | Optional experimental ₹200 Logistic Regression bundle and metadata. |
| `models/*.json` | Human-readable model/training metadata; does not replace the required model binaries. |

The model files must match the preprocessing and model architecture used by
this code. Retrain or obtain compatible artifacts if files are missing or
incompatible.

## Known limitations and safe use

- The project analyzes visible pixels only. It cannot inspect raised print,
  paper feel, UV fluorescence, infrared response, watermarks under transmitted
  light, or angle-dependent ink effects.
- Results depend on the quality, denomination, side, orientation, and
  coverage of the available reference photos. A photo that is rotated 180°,
  cropped, blurred, reflective, or poorly lit can be inconclusive or
  misleading.
- ORB, SSIM, OCR, anomaly scores, and the experimental classifier are
  evidence about the supplied images, not proof of authenticity.
- OCR only reports detected text; it does not validate serial numbers,
  denomination, or security features.
- Small or repeated training sets can overfit. Images of the same physical
  note must not be split across training and evaluation sets.
- Do not use the result as the sole basis for accepting or rejecting currency.
  Seek physical verification from a bank or qualified authority.

## Troubleshooting

- **Model files missing:** run the genuine-reference training workflow and
  check that all three shared model files listed above were saved under
  `models/`.
- **No matching reference / Unknown denomination:** verify that
  `data/references/` contains a readable same-denomination image and that the
  denomination appears in its filename.
- **Boundary not detected:** capture the entire note on a plain contrasting
  background with the note upright, in focus, and without strong glare.
- **Upload rejected:** only readable JPG/PNG images up to 12 MB are accepted.
- **Camera unavailable:** allow browser camera permission and use
  `localhost` or HTTPS.
- **OCR unavailable:** install the requirements and check the server output;
  OCR is supplementary and the scan may still return a result without it.
