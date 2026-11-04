
"""
SmartRupee - Genuine Reference Anomaly Model Training

Uses MobileNetV2 to extract visual features and an autoencoder
to learn the distribution of genuine reference images.

IMPORTANT:
This is NOT a binary counterfeit classifier.
It checks visual consistency with genuine reference images.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

SIZE = (224, 224)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

DENOMINATIONS = ("10", "20", "50", "100", "200", "500", "2000")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def image_files(folder):

    folder = Path(folder)

    if not folder.exists():
        return []

    result = []

    for path in sorted(folder.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(path))

        if image is not None:
            result.append(path)

    return result


# ============================================================
# DENOMINATION DETECTION
# ============================================================

def detect_denomination(path):

    """
    Detect denomination from filename or folder structure.

    Examples:
        100_front.jpg
        rs500_note.png
        genuine/200/image1.jpg

    Avoids confusing 100 with 10.
    """

    text = str(path).lower()

    patterns = [
        (r"(?<!\d)2000(?!\d)", "2000"),
        (r"(?<!\d)500(?!\d)", "500"),
        (r"(?<!\d)200(?!\d)", "200"),
        (r"(?<!\d)100(?!\d)", "100"),
        (r"(?<!\d)50(?!\d)", "50"),
        (r"(?<!\d)20(?!\d)", "20"),
        (r"(?<!\d)10(?!\d)", "10"),
    ]

    matches = []

    for pattern, denomination in patterns:

        if re.search(pattern, text):
            matches.append(denomination)

    if len(matches) == 1:
        return matches[0]

    return None


def filter_denomination(paths, denomination):

    if denomination is None:
        return paths

    selected = []

    for path in paths:

        detected = detect_denomination(path)

        if detected == denomination:
            selected.append(path)

    return selected


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_rgb(path, align_to_note=False):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    # Resize large images.
    height, width = image.shape[:2]

    scale = min(
        1.0,
        1600 / max(height, width)
    )

    if scale < 1:

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # Attempt note alignment.
    # If boundary detection fails, keep the original image.
    if align_to_note:

        try:

            from smartrupee.engine import detect_note

            aligned, found, _ = detect_note(image)

            if found and aligned is not None:
                image = aligned

        except Exception as error:

            print(
                f"Alignment skipped for {Path(path).name}: "
                f"{error}"
            )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        SIZE,
        interpolation=cv2.INTER_AREA
    )

    return image.astype("float32")


# ============================================================
# MOBILENETV2 FEATURE EXTRACTOR
# ============================================================

def build_extractor():

    base = keras.applications.MobileNetV2(
        input_shape=SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

    inputs = keras.Input(
        shape=SIZE + (3,)
    )

    x = keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )

    x = base(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D(
        name="reference_embedding"
    )(x)

    model = keras.Model(
        inputs,
        x,
        name="smartrupee_reference_extractor"
    )

    return model


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augmented_image(image, rng):

    image = image.astype("float32")

    angle = rng.uniform(-3, 3)

    scale = rng.uniform(0.97, 1.03)

    matrix = cv2.getRotationMatrix2D(
        (SIZE[0] / 2, SIZE[1] / 2),
        angle,
        scale
    )

    result = cv2.warpAffine(
        image,
        matrix,
        SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    brightness = rng.uniform(-8, 8)
    contrast = rng.uniform(0.95, 1.05)

    result = (
        (result - 127.5) * contrast
        + 127.5
        + brightness
    )

    return np.clip(
        result,
        0,
        255
    ).astype("float32")


# ============================================================
# ANOMALY AUTOENCODER
# ============================================================

def build_anomaly_model(input_dimension):

    latent_dim = min(
        128,
        max(16, input_dimension // 4)
    )

    inputs = keras.Input(
        shape=(input_dimension,)
    )

    encoded = layers.Dense(
        latent_dim,
        activation="relu"
    )(inputs)

    decoded = layers.Dense(
        input_dimension,
        activation="linear"
    )(encoded)

    model = keras.Model(
        inputs,
        decoded,
        name="genuine_reference_anomaly_autoencoder"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="mse"
    )

    return model


# ============================================================
# TRAINING FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train SmartRupee genuine-reference model"
    )

    parser.add_argument(
        "--data",
        default="dataset",
        help="Dataset directory"
    )

    parser.add_argument(
        "--genuine-dir",
        default=None,
        help="Folder containing genuine reference images"
    )

    parser.add_argument(
        "--denomination",
        choices=DENOMINATIONS,
        default=None,
        help="Train a model for one denomination"
    )

    parser.add_argument(
        "--skip-denomination-filter",
        action="store_true",
        help="Use all images in genuine-dir without filename filtering"
    )

    parser.add_argument(
        "--min-genuine",
        type=int,
        default=5,
        help="Minimum number of original genuine images"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # --------------------------------------------------------
    # SELECT DATASET
    # --------------------------------------------------------

    if args.genuine_dir:

        genuine_folder = Path(args.genuine_dir)

        if not genuine_folder.is_absolute():
            genuine_folder = PROJECT_DIR / genuine_folder

    else:

        genuine_folder = (
            PROJECT_DIR /
            args.data /
            "train" /
            "genuine"
        )

    if not genuine_folder.exists():

        raise FileNotFoundError(
            f"Genuine folder not found: {genuine_folder}"
        )

    all_images = image_files(genuine_folder)

    print("\n======================================")
    print("       SMART RUPEE TRAINING")
    print("======================================")

    print("\nGenuine folder:", genuine_folder)
    print("Readable images:", len(all_images))

    # --------------------------------------------------------
    # FILTER IMAGES
    # --------------------------------------------------------

    if args.denomination and not args.skip_denomination_filter:

        genuine = filter_denomination(
            all_images,
            args.denomination
        )

    else:

        genuine = all_images

    print("Selected images:", len(genuine))

    if args.denomination:
        print("Denomination: Rs.", args.denomination)

    if len(genuine) < args.min_genuine:

        raise ValueError(
            "\nInsufficient genuine images.\n"
            f"Required: {args.min_genuine}\n"
            f"Found: {len(genuine)}\n"
            "Check your filenames/folders or use "
            "--skip-denomination-filter for a folder "
            "containing only the selected denomination."
        )

    # --------------------------------------------------------
    # MODEL PATHS
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        f"_{args.denomination}"
        if args.denomination
        else ""
    )

    extractor_path = (
        MODEL_DIR /
        "genuine_feature_extractor.keras"
    )

    anomaly_path = (
        MODEL_DIR /
        f"genuine_reference_anomaly{suffix}.keras"
    )

    distribution_path = (
        MODEL_DIR /
        f"genuine_reference_distribution{suffix}.npz"
    )

    summary_path = (
        MODEL_DIR /
        f"genuine_reference_distribution{suffix}.json"
    )

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    print("\nLoading images...")

    images = []
    loaded_files = []

    for index, path in enumerate(genuine, start=1):

        try:

            image = load_rgb(
                path,
                align_to_note=bool(args.denomination)
            )

            images.append(image)
            loaded_files.append(path)

            print(
                f"[{index}/{len(genuine)}] "
                f"Loaded {path.name}"
            )

        except Exception as error:

            print(
                f"Skipped {path.name}: {error}"
            )

    if len(images) < args.min_genuine:

        raise ValueError(
            "Not enough successfully loaded images."
        )

    images = np.stack(images).astype("float32")

    print("\nImage array:", images.shape)

    # --------------------------------------------------------
    # BUILD FEATURE EXTRACTOR
    # --------------------------------------------------------

    print("\nLoading MobileNetV2...")

    if extractor_path.exists():

        extractor = keras.models.load_model(
            str(extractor_path),
            compile=False
        )

        print("Existing feature extractor loaded.")

    else:

        extractor = build_extractor()

        print("New feature extractor created.")

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    print("\nGenerating augmented images...")

    rng = np.random.default_rng(args.seed)

    augmented = []

    for image in images:

        for _ in range(3):

            augmented.append(
                augmented_image(image, rng)
            )

    augmented = np.stack(augmented).astype("float32")

    print("Original:", len(images))
    print("Augmented:", len(augmented))

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    print("\nExtracting original features...")

    embeddings = extractor.predict(
        images,
        batch_size=8,
        verbose=1
    ).astype("float32")

    print("\nExtracting augmented features...")

    augmented_embeddings = extractor.predict(
        augmented,
        batch_size=8,
        verbose=1
    ).astype("float32")

    all_embeddings = np.vstack([
        embeddings,
        augmented_embeddings
    ]).astype("float32")

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    mean = all_embeddings.mean(
        axis=0
    ).astype("float32")

    scale = np.maximum(
        all_embeddings.std(axis=0),
        1e-5
    ).astype("float32")

    standardized = (
        all_embeddings - mean
    ) / scale

    # --------------------------------------------------------
    # TRAIN AUTOENCODER
    # --------------------------------------------------------

    print("\nTraining anomaly model...")

    anomaly = build_anomaly_model(
        standardized.shape[1]
    )

    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=7,
            restore_best_weights=True
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )

    ]

    anomaly.fit(
        standardized,
        standardized,
        epochs=args.epochs,
        batch_size=min(16, len(standardized)),
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # CALCULATE THRESHOLDS
    # --------------------------------------------------------

    print("\nCalculating thresholds...")

    original_standardized = (
        embeddings - mean
    ) / scale

    reconstructed = anomaly.predict(
        original_standardized,
        batch_size=8,
        verbose=0
    )

    reconstruction_errors = np.mean(
        np.square(
            reconstructed - original_standardized
        ),
        axis=1
    )

    reference_distances = np.sqrt(
        np.mean(
            np.square(original_standardized),
            axis=1
        )
    )

    reconstruction_threshold = float(
        max(
            np.percentile(reconstruction_errors, 95) * 2.0,
            np.mean(reconstruction_errors)
            + 3 * np.std(reconstruction_errors),
            1e-6
        )
    )

    distance_threshold = float(
        max(
            np.percentile(reference_distances, 99) * 1.25,
            np.mean(reference_distances)
            + 3 * np.std(reference_distances),
            1e-6
        )
    )

    print(
        "Reconstruction threshold:",
        reconstruction_threshold
    )

    print(
        "Distance threshold:",
        distance_threshold
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    print("\nSaving models...")

    extractor.save(
        str(extractor_path)
    )

    anomaly.save(
        str(anomaly_path)
    )

    np.savez(
        distribution_path,

        mean=mean,

        scale=scale,

        reference_embeddings=embeddings,

        reconstruction_threshold=np.array(
            [reconstruction_threshold],
            dtype="float32"
        ),

        distance_threshold=np.array(
            [distance_threshold],
            dtype="float32"
        )
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    summary = {

        "model_type": "genuine_reference_anomaly_detection",

        "base_model": "MobileNetV2 ImageNet",

        "denomination": args.denomination,

        "genuine_reference_count": len(images),

        "augmented_image_count": len(augmented),

        "embedding_dimension": int(
            embeddings.shape[1]
        ),

        "image_size": list(SIZE),

        "epochs_requested": args.epochs,

        "reconstruction_threshold": reconstruction_threshold,

        "distance_threshold": distance_threshold,

        "reference_files": [
            str(path) for path in loaded_files
        ],

        "counterfeit_detection_claim": False,

        "interpretation": (
            "The model checks visual consistency with available "
            "genuine-reference images. An unusual result does not "
            "prove that a note is counterfeit. A consistent result "
            "does not certify authenticity."
        )
    }

    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # VERIFY SAVED FILES
    # --------------------------------------------------------

    print("\nVerifying saved models...")

    keras.models.load_model(
        str(extractor_path),
        compile=False
    )

    keras.models.load_model(
        str(anomaly_path),
        compile=False
    )

    with np.load(distribution_path) as data:

        required = {
            "mean",
            "scale",
            "reference_embeddings",
            "reconstruction_threshold",
            "distance_threshold"
        }

        missing = required.difference(data.files)

        if missing:

            raise ValueError(
                f"Missing distribution fields: {missing}"
            )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n======================================")
    print("      TRAINING COMPLETED")
    print("======================================")

    print("\nSaved files:")

    print(extractor_path)
    print(anomaly_path)
    print(distribution_path)
    print(summary_path)

    print("\nGenuine images:", len(images))
    print("Augmented images:", len(augmented))

    print("\nAll model files verified successfully.")

    print(
        "\nReminder: This is a genuine-reference anomaly model, "
        "not a counterfeit classifier."
    )


if __name__ == "__main__":
    main()
