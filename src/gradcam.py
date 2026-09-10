"""
Grad-CAM: highlights which regions of an image most influenced the
model's prediction. Great for interpretability and for making your
demo app / resume video more compelling.
"""
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image batch, shape (1, H, W, 3)
    model: the full Keras model
    last_conv_layer_name: name of the last conv layer (e.g. 'Conv_1' for MobileNetV2,
                           check model.summary() to confirm the exact name)
    """
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img_rgb, heatmap, alpha=0.4):
    """
    original_img_rgb: np.array (H, W, 3), values 0-255, RGB
    heatmap: 2D np.array output from make_gradcam_heatmap
    Returns an RGB np.array with the heatmap overlaid.
    """
    heatmap = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = heatmap_color * alpha + original_img_rgb * (1 - alpha)
    return overlaid.astype(np.uint8)
