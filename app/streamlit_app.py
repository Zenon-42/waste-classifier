"""
Streamlit demo app: upload an image, get the predicted waste category,
confidence, disposal tip, and a Grad-CAM visualization.

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = "models/mobilenet_finetuned_best.keras"
CLASSES_PATH = "models/mobilenet_finetuned_classes.json"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "Conv_1"  # confirm via model.summary() if this differs

DISPOSAL_TIPS = {
    "cardboard": "Flatten and place in the recyclable/dry waste bin.",
    "glass": "Rinse and place in the recyclable bin. Handle broken glass separately.",
    "metal": "Rinse cans/foil and place in the recyclable bin.",
    "paper": "Place in the recyclable/dry waste bin; avoid soiled paper.",
    "plastic": "Rinse and check the recycling code; place in the recyclable bin.",
    "trash": "Non-recyclable — place in general waste.",
}


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image)
    batch = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr, batch


def main():
    st.set_page_config(page_title="Waste Segregation Classifier", page_icon="♻️")
    st.title("♻️ Waste Segregation Classifier")
    st.write("Upload a photo of a waste item and get its category + disposal tip.")

    model, class_names = load_model_and_classes()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        arr, batch = preprocess(image)
        preds = model.predict(batch)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = float(preds[pred_idx])

        st.subheader(f"Prediction: **{pred_class.capitalize()}** ({confidence:.1%} confidence)")
        st.info(DISPOSAL_TIPS.get(pred_class, "Dispose according to local guidelines."))

        with st.expander("See prediction breakdown"):
            for cls, prob in sorted(zip(class_names, preds), key=lambda x: -x[1]):
                st.write(f"{cls}: {prob:.1%}")

        with st.expander("See Grad-CAM (what the model focused on)"):
            heatmap, _ = make_gradcam_heatmap(batch, model, LAST_CONV_LAYER, pred_idx)
            overlay = overlay_heatmap(arr, heatmap)
            st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)


if __name__ == "__main__":
    main()
