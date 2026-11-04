"""Train the experimental user-labeled ₹200 image classifier."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from smartrupee.engine import MODEL_PATH
from train import image_files, load_rgb


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH_200 = MODEL_DIR / "currency_authenticity_200.joblib"
SUMMARY_PATH_200 = MODEL_DIR / "currency_authenticity_200.json"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an experimental ₹200 classifier from user-labeled "
            "genuine and fake image folders."
        )
    )
    parser.add_argument(
        "--genuine-dir",
        default="data/training_200/genuine",
        help="Folder of user-labeled genuine ₹200 photos",
    )
    parser.add_argument(
        "--fake-dir",
        default="data/training_200/fake",
        help="Folder of user-labeled fake ₹200 photos",
    )
    parser.add_argument("--min-per-class", type=int, default=3)
    args = parser.parse_args()

    genuine_dir = Path(args.genuine_dir)
    fake_dir = Path(args.fake_dir)
    if not genuine_dir.is_absolute():
        genuine_dir = PROJECT_DIR / genuine_dir
    if not fake_dir.is_absolute():
        fake_dir = PROJECT_DIR / fake_dir

    if args.min_per_class < 2:
        raise ValueError("--min-per-class must be at least 2.")

    genuine_files = image_files(genuine_dir)
    fake_files = image_files(fake_dir)
    if len(genuine_files) < args.min_per_class:
        raise ValueError(
            f"Need at least {args.min_per_class} readable genuine images; "
            f"found {len(genuine_files)} in {genuine_dir}."
        )
    if len(fake_files) < args.min_per_class:
        raise ValueError(
            f"Need at least {args.min_per_class} readable fake images; "
            f"found {len(fake_files)} in {fake_dir}."
        )

    extractor_path = Path(MODEL_PATH)
    if not extractor_path.is_file():
        raise FileNotFoundError(
            f"MobileNetV2 feature extractor not found: {extractor_path}"
        )
    extractor = keras.models.load_model(extractor_path, compile=False)

    paths = genuine_files + fake_files
    labels = np.asarray(
        ["genuine"] * len(genuine_files) + ["fake"] * len(fake_files)
    )
    images = np.stack([
        load_rgb(path, align_to_note=True)
        for path in paths
    ])
    embeddings = extractor.predict(images, batch_size=8, verbose=1)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    classifier.fit(embeddings, labels)
    training_accuracy = float(classifier.score(embeddings, labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": "experimental_user_labeled_200_image_classifier",
        "denomination": "200",
        "classes": ["fake", "genuine"],
        "class_counts": {
            "genuine": len(genuine_files),
            "fake": len(fake_files),
        },
        "training_images": {
            "genuine": [path.name for path in genuine_files],
            "fake": [path.name for path in fake_files],
        },
        "training_accuracy": training_accuracy,
        "training_accuracy_is_not_validation": True,
        "validation_status": (
            "No independent-note validation set was supplied. Images may "
            "show the same physical note or repeated specimen designs."
        ),
        "label_source": (
            "Labels were supplied by the user and were not independently "
            "authenticated."
        ),
        "decision_thresholds": {
            "fake_score_minimum": 0.60,
            "genuine_score_minimum": 0.60,
        },
        "score_warning": (
            "Logistic-regression scores are uncalibrated and are not "
            "probabilities of authenticity."
        ),
    }

    joblib.dump(
        {
            "classifier": classifier,
            "metadata": metadata,
        },
        MODEL_PATH_200,
    )
    SUMMARY_PATH_200.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print(f"Genuine training images: {len(genuine_files)}")
    print(f"Fake training images: {len(fake_files)}")
    print(f"Training-set fit accuracy (not validation): {training_accuracy:.3f}")
    print(f"Saved classifier: {MODEL_PATH_200}")
    print(f"Saved training summary: {SUMMARY_PATH_200}")
    print(
        "This prototype has no independent-note validation and must not be "
        "used as proof of authenticity."
    )


if __name__ == "__main__":
    main()
