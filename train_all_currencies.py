"""Train a 12-class SmartRupee model: genuine/fake x six denominations.

Expected layout:
 dataset/all_currencies/
   train/{genuine_10,fake_10,...,genuine_500,fake_500}/image files
   validation/{same 12 class folders}/image files
   test/{same 12 class folders}/image files  # optional, held out until final evaluation

Use separate physical banknotes in each split. Do not place alternate photos of
one physical note in more than one split. This is an experimental visual model,
not an official banknote authentication system.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DENOMINATIONS = ("10", "20", "50", "100", "200", "500")
CLASS_NAMES = [f"{kind}_{denom}" for denom in DENOMINATIONS for kind in ("genuine", "fake")]
IMG_SIZE = (224, 224)


def load_split(root: Path, split: str, batch: int):
    folder = root / split
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    missing = [c for c in CLASS_NAMES if not (folder / c).is_dir()]
    if missing:
        raise ValueError(f"Missing class folders under {folder}: {', '.join(missing)}")
    ds = keras.utils.image_dataset_from_directory(
        folder, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
        image_size=IMG_SIZE, batch_size=batch, shuffle=(split == "train"),
        seed=42)
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for _, labels in ds:
        counts += np.bincount(labels.numpy(), minlength=len(CLASS_NAMES))
    empty = [CLASS_NAMES[i] for i, n in enumerate(counts) if n == 0]
    if empty:
        raise ValueError(f"No readable images in these {split} classes: {', '.join(empty)}")
    return ds, counts


def make_model(weights: str | None):
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="currency_image")
    # Do not horizontally flip banknotes: mirrored text/security features are unrealistic.
    aug = keras.Sequential([
        layers.RandomRotation(0.025),
        layers.RandomTranslation(0.04, 0.04),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.12),
    ], name="safe_augmentation")
    x = aug(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False,
        weights=weights, pooling="avg")
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(x)
    model = keras.Model(inputs, outputs)
    return model, backbone


def main():
    parser = argparse.ArgumentParser(description="Train SmartRupee for ₹10/20/50/100/200/500")
    parser.add_argument("--data-dir", default="dataset/all_currencies", help="Contains train, validation, optional test")
    parser.add_argument("--output-dir", default="models/all_currencies")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet", help="Use none if offline; ImageNet weights may download on first run")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze last 30 backbone layers after initial training")
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive; fine-tune-epochs cannot be negative")

    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ds, train_counts = load_split(root, "train", args.batch_size)
    val_ds, val_counts = load_split(root, "validation", args.batch_size)
    print("\nTraining image counts:")
    for c, n in zip(CLASS_NAMES, train_counts): print(f"  {c:12s} {n}")
    print("\nValidation image counts:")
    for c, n in zip(CLASS_NAMES, val_counts): print(f"  {c:12s} {n}")
    if min(train_counts) < 1:
        raise ValueError("Each of the 12 training classes needs at least 1 readable image.")
    if min(val_counts) < 1:
        raise ValueError("Each validation class needs at least 1 readable image.")

    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=labels)
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    model, backbone = make_model(None if args.weights == "none" else "imagenet")
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(out / "best_all_currencies.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    print("\nPhase 1: train classification head")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weights, callbacks=callbacks)

    if args.fine_tune and args.fine_tune_epochs:
        print("\nPhase 2: fine-tune last 30 MobileNetV2 layers")
        backbone.trainable = True
        for layer in backbone.layers[:-30]: layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                  class_weight=class_weights, callbacks=callbacks)

    best_path = out / "best_all_currencies.keras"
    best = keras.models.load_model(best_path)
    best.save(out / "smart rupee all currencies.keras")
    (out / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    (out / "training_config.json").write_text(json.dumps({
        "denominations": list(DENOMINATIONS), "classes": CLASS_NAMES,
        "image_size": list(IMG_SIZE), "backbone": "MobileNetV2",
        "weights": args.weights, "train_counts": dict(zip(CLASS_NAMES, map(int, train_counts))),
        "validation_counts": dict(zip(CLASS_NAMES, map(int, val_counts))),
        "warning": "Experimental visual classification only; not official authentication."
    }, indent=2), encoding="utf-8")

    if (root / "test").is_dir():
        test_ds, test_counts = load_split(root, "test", args.batch_size)
        y_true = np.concatenate([y.numpy() for _, y in test_ds])
        probs = best.predict(test_ds, verbose=1)
        y_pred = probs.argmax(axis=1)
        report = classification_report(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)),
                                       target_names=CLASS_NAMES, zero_division=0, output_dict=True)
        (out / "test_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "test_confusion_matrix.json").write_text(json.dumps(confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES))).tolist(), indent=2), encoding="utf-8")
        print("\nHeld-out test accuracy:", float(np.mean(y_true == y_pred)))
        print("Saved test report and confusion matrix in", out)
    print("\nTraining complete. Model saved in:", out)
    print("Do not use the output as proof that a banknote is genuine or counterfeit.")

if __name__ == "__main__":
    main()
