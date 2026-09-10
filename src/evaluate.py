"""
Loads a trained model and reports confusion matrix + per-class metrics
on the test set. This is the "error analysis" step that makes the
project stand out beyond just a headline accuracy number.

Usage:
    python -m src.evaluate --model-path models/mobilenet_finetuned_best.keras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import get_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(args.model_path)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    out_path = Path(args.model_path).parent / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"\nConfusion matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
