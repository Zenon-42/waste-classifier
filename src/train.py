"""
CLI training script.

Examples:
    python -m src.train --model baseline --epochs 15
    python -m src.train --model mobilenet --epochs 10 --fine-tune
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data_loader import get_datasets
from src.model import build_baseline_cnn, build_mobilenet_model

MODELS_DIR = Path("models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "mobilenet"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune", action="store_true",
                         help="Only used with --model mobilenet: unfreezes top layers")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    if args.model == "baseline":
        model = build_baseline_cnn(num_classes)
        run_name = "baseline_cnn"
    else:
        model = build_mobilenet_model(num_classes, fine_tune=args.fine_tune)
        run_name = "mobilenet_finetuned" if args.fine_tune else "mobilenet_frozen"

    model.summary()

    checkpoint_path = MODELS_DIR / f"{run_name}_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save class names alongside the model so evaluate.py / app can load them
    with open(MODELS_DIR / f"{run_name}_classes.json", "w") as f:
        json.dump(class_names, f)

    # Quick test-set check
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    with open(MODELS_DIR / f"{run_name}_history.json", "w") as f:
        json.dump(history.history, f)

    print(f"\nBest model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
