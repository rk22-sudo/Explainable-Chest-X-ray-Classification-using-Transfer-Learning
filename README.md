# Capstone: Medical Image Classification (X-ray) with Transfer Learning + Grad-CAM

**Subject:** Deep Learning and Applications

## What this project does
1. Loads chest X-ray images (COVID-19 Radiography Database — COVID / Normal / Lung Opacity / Viral Pneumonia).
2. Fine-tunes a pretrained **ResNet18** (transfer learning) to classify the 4 classes.
3. Evaluates with accuracy, a classification report, and a confusion matrix.
4. Implements **Grad-CAM** (from scratch, no extra dependency) to highlight the image regions
   driving each prediction — the "explainable AI" piece, so a clinician could sanity-check the model.

## Files
- `medical_image_classification_capstone.ipynb` — the whole project, one notebook, top to bottom.
- `requirements.txt` — Python packages needed if running locally instead of Colab.

## How to run

### Option A — Google Colab (recommended)
1. Go to https://colab.research.google.com and upload `medical_image_classification_capstone.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run all cells top to bottom. The notebook downloads the dataset itself via `kagglehub`
   (you'll need a free Kaggle account/API token — Colab will prompt you, or run
   `kagglehub.login()` if it asks).
4. Training a 300-images-per-class subset for 5 epochs takes a few minutes on the free GPU.

### Option B — Your laptop (CPU only)
Works, but slower. To keep it quick:
```bash
pip install -r requirements.txt
jupyter notebook medical_image_classification_capstone.ipynb
```
In the notebook, lower `N_PER_CLASS` (e.g. to 100) and keep `EPOCHS` at 3–5 so a full run finishes
in a reasonable time on CPU. Your i5-1235U/16GB setup can comfortably do this for a demo-scale run;
it just won't be fast enough for full-dataset, many-epoch training — that's what Colab's GPU is for.

## Notes for your report/viva
- **Why ResNet18 and not something bigger?** Small backbone = fast to fine-tune, still strong on
  medical imaging with transfer learning, and realistic for a capstone timeline/hardware.
- **Why freeze most layers?** The early conv layers already encode general features (edges,
  textures) from ImageNet; only the last block + classifier need retraining for X-rays, which
  cuts training time drastically and reduces overfitting on a small medical dataset.
- **Why Grad-CAM?** It's a standard XAI method for CNNs — no architecture changes needed, and it
  directly answers "what part of the X-ray made the model say COVID?", which is exactly what a
  clinician would want to verify before trusting the model's output.
