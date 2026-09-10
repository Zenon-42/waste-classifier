"""
Handles splitting the raw dataset into train/val/test folders and
building tf.data pipelines for training.
"""
import argparse
import shutil
import random
from pathlib import Path

import tensorflow as tf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data/raw/<class>/*.jpg into data/processed/{train,val,test}/<class>/*.jpg"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(SEED)
    classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for split in ["train", "val", "test"]:
        for cls in classes:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        images = list((RAW_DIR / cls).glob("*.*"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in splits.items():
            for f in files:
                shutil.copy(f, PROCESSED_DIR / split / cls / f.name)

        print(f"{cls}: {n} total -> train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    print("Done. Processed data at:", PROCESSED_DIR.resolve())


def get_datasets():
    """Returns (train_ds, val_ds, test_ds, class_names) as tf.data pipelines."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=False
    )

    class_names = train_ds.class_names

    # Performance: cache + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Split raw data into train/val/test")
    args = parser.parse_args()

    if args.split:
        split_dataset()
