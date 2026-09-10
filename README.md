# Waste Segregation Classifier

A CNN-based image classifier that sorts waste into categories (cardboard, glass,
metal, paper, plastic, trash) using transfer learning (MobileNetV2), with
Grad-CAM interpretability and a Streamlit demo app.

## Project Structure

```
waste-classifier/
├── data/                   # dataset goes here (not committed to git)
│   ├── raw/                # original downloaded images, organized by class folder
│   └── processed/          # train/val/test split (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # dataset splitting + tf.data pipelines
│   ├── model.py             # baseline CNN + transfer learning model definitions
│   ├── train.py              # training script (CLI)
│   ├── evaluate.py          # confusion matrix, per-class metrics
│   └── gradcam.py            # Grad-CAM visualization utility
├── models/                  # saved trained models (.keras / .h5)
├── notebooks/
│   └── eda.ipynb            # exploratory data analysis
├── app/
│   └── streamlit_app.py     # deployable demo app
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download the Kaggle "Garbage Classification" dataset and place it under
`data/raw/` so it looks like:

```
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Then run the split:
```bash
python -m src.data_loader --split
```

## Training

```bash
# Baseline CNN from scratch
python -m src.train --model baseline --epochs 15

# Transfer learning (recommended main model)
python -m src.train --model mobilenet --epochs 10 --fine-tune
```

## Evaluation

```bash
python -m src.evaluate --model-path models/mobilenet_best.keras
```

## Run the demo app

```bash
streamlit run app/streamlit_app.py
```

## Roadmap
- [ ] Baseline CNN
- [ ] MobileNetV2 transfer learning
- [ ] Grad-CAM visualizations
- [ ] Streamlit app
- [ ] Deploy to Streamlit Community Cloud / HuggingFace Spaces
- [ ] (Stretch) Export to TFLite for edge deployment
