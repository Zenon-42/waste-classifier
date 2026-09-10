"""
Model definitions: a baseline CNN trained from scratch, and a
MobileNetV2-based transfer learning model.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)

# Simple data augmentation applied at training time
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")


def build_baseline_cnn(num_classes: int) -> tf.keras.Model:
    """Small CNN trained from scratch — serves as your baseline to beat."""
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_model(num_classes: int, fine_tune: bool = False) -> tf.keras.Model:
    """MobileNetV2 transfer learning model. This is your main/best model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = fine_tune
    if fine_tune:
        # Only unfreeze the last ~30 layers to avoid destroying pretrained features
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenet_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 if fine_tune else 1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
