"""
Grad-CAM: highlights which regions of an image most influenced the
model's prediction. Great for interpretability and for making your
demo app / resume video more compelling.
"""
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, pred_index=None,
                          base_model_name="mobilenetv2_1.00_224",
                          conv_layer_name="Conv_1",
                          pool_layer_name="global_average_pooling2d",
                          dense_layer_name="dense"):
    """
    img_array: RAW (un-preprocessed) image batch, shape (1, H, W, 3), values 0-255
    model: the full Keras model (base_model_name/etc. must match your model's
           actual layer names — check with `for l in model.layers: print(l.name)`)

    Why this looks different from a "normal" Grad-CAM implementation:
    MobileNetV2 is nested inside our model as a single black-box layer, so its
    internal Conv_1 layer isn't directly reachable via model.get_layer(). We
    reach into the nested model for the conv output, then manually replicate
    the pooling + dense steps that come after it, so gradients can flow
    through the whole chain inside one GradientTape.
    """
    base_model = model.get_layer(base_model_name)
    dense_layer = model.get_layer(dense_layer_name)

    # Sub-model that exposes both the last conv activations AND the base
    # model's final feature output (needed to continue the forward pass)
    conv_model = tf.keras.Model(
        base_model.input,
        [base_model.get_layer(conv_layer_name).output, base_model.output]
    )

    # Replicate MobileNetV2's expected preprocessing (the outer model does
    # this via tf.keras.applications.mobilenet_v2.preprocess_input, which is
    # a plain function call, not a layer — so it doesn't show up in
    # model.layers and we must reapply it manually here)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_array, tf.float32)
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_features = conv_model(preprocessed)
        tape.watch(conv_outputs)  # not a trainable Variable, so watch explicitly

        # Manually replicate GlobalAveragePooling2D -> Dense(softmax)
        pooled = tf.reduce_mean(base_features, axis=(1, 2))
        predictions = dense_layer(pooled)

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