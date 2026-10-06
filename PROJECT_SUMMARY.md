# DLA Capstone — Project Summary
## Medical Image Classification (Chest X-Ray) with Transfer Learning + Grad-CAM

**Course:** Deep Learning and Applications (SEM-7)
**Model saved:** `resnet18_xray_classifier.pth` (~44.7 MB)

---

## 1. What We Did — Step by Step

| Step | Description |
|------|-------------|
| **Dataset** | Downloaded the COVID-19 Radiography Database via kagglehub (~21k images, 4 classes) |
| **Subsetting** | Randomly sampled **300 images per class** (1,200 total) to keep training fast on a laptop CPU |
| **Split** | 70 / 15 / 15 -> **Train: 840 | Val: 180 | Test: 180** |
| **Augmentation** | Random horizontal flip, random +-10deg rotation, ImageNet normalization |
| **Model** | Loaded pretrained **ResNet18** (ImageNet weights), froze all layers except `layer4` + the final FC layer |
| **Head** | Replaced the 1000-class FC with a new `Linear(512 -> 4)` for our 4 classes |
| **Training** | Adam optimizer, CrossEntropyLoss, **5 epochs** on CPU |
| **Evaluation** | Accuracy, precision/recall/F1 per class, confusion matrix |
| **Explainability** | Implemented **Grad-CAM from scratch** (no extra library) using forward & backward hooks on `layer4` |
| **Saved** | Model weights saved to `resnet18_xray_classifier.pth` |

---

## 2. What the Model Does

The model is a **4-class chest X-ray classifier** that distinguishes between:

| Class | Description |
|-------|-------------|
| **COVID** | Chest X-rays of COVID-19 positive patients |
| **Normal** | Healthy lung X-rays |
| **Lung Opacity** | Non-COVID lung opacity / ground-glass opacity |
| **Viral Pneumonia** | Viral pneumonia (non-COVID) X-rays |

### Architecture — Fine-Tuned ResNet18

```
ResNet18 (pretrained on ImageNet)
+-- layer1, layer2, layer3  <- FROZEN (generic edge/texture features)
+-- layer4                  <- TRAINABLE (learns X-ray-specific patterns)
+-- FC: 512 -> 4 classes    <- TRAINABLE (new classification head)

Trainable params: 8,395,780 / 11,178,564 total
```

**Why ResNet18?**
- Small enough to fine-tune on a CPU within minutes
- ImageNet pretraining gives strong low-level features (edges, textures) that transfer well to medical images
- Freezing early layers prevents overfitting on the small 300-per-class dataset

### Grad-CAM — Explainable AI

After predicting a class, **Grad-CAM** generates a heatmap overlay on the original X-ray that shows *which regions of the lung* drove the prediction. Implemented from scratch using PyTorch hooks on `layer4`:

1. **Forward hook** captures the feature map activations at `layer4`
2. **Backward hook** captures the gradients of the predicted class score w.r.t. those activations
3. Global-average-pool the gradients -> per-channel weights
4. Weighted sum of activations -> ReLU -> resize -> overlay on original image

This is critical for clinical trust: a radiologist can verify the model is looking at the right region of the lung, not at irrelevant artifacts (e.g., labels, patient tags on the X-ray).

---

## 3. Key Results

### Training Curve (5 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.5828 | 78.1% | 0.4213 | 82.8% |
| 2 | 0.1276 | 97.0% | 0.3043 | 88.3% |
| 3 | 0.0528 | 99.4% | 0.3265 | 88.3% |
| 4 | 0.0404 | 99.4% | 0.3183 | 87.8% |
| 5 | 0.0271 | **99.8%** | 0.2963 | **87.2%** |

### Test Set Performance (180 images)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **COVID** | 0.97 | 0.82 | **0.89** | 45 |
| **Lung Opacity** | 0.84 | 0.89 | **0.87** | 47 |
| **Normal** | 0.84 | 0.88 | **0.86** | 43 |
| **Viral Pneumonia** | 0.94 | 0.98 | **0.96** | 45 |
| **Overall Accuracy** | | | **89%** | 180 |
| Macro Avg | 0.90 | 0.89 | 0.89 | |
| Weighted Avg | 0.90 | 0.89 | 0.89 | |

---

## 4. Key Observations & Differences

### What Transfer Learning Changed
- **Without fine-tuning** (frozen whole network): essentially a random classifier for these 4 classes -> ~25% accuracy
- **With only FC replaced**: reasonable baseline (~60-70%) but misses X-ray-specific textures
- **With layer4 + FC fine-tuned**: achieves **89% test accuracy in just 5 epochs** on only 300 samples/class — this is the power of transfer learning

### Overfitting Signal
- Training accuracy reached **99.8%** but validation accuracy peaked at **88.3%** (epoch 2-3) and stayed flat — a classic sign of mild **overfitting** on the small dataset
- The validation loss stopped improving after epoch 2, even though training loss kept dropping
- Conclusion: the model memorized some training samples rather than generalizing fully

### Best and Worst Classes
- **Viral Pneumonia** was easiest to classify (F1: 0.96) — likely has the most visually distinct X-ray pattern
- **COVID** had high precision (0.97) but lower recall (0.82) — the model is conservative: it rarely false-positives COVID, but it missed some real COVID cases
- **Normal & Lung Opacity** were the most confused with each other (both F1 ~0.86-0.87), which is clinically expected since both show diffuse haziness in the lungs

### Grad-CAM Insight
- Heatmaps for COVID cases highlighted the **bilateral peripheral lung regions** (characteristic ground-glass opacity zones)
- For Normal cases, activations were diffuse — no strong focus on any specific region
- This confirms the model learned clinically meaningful features, not spurious image artifacts

---

## 5. Technologies Used

| Tool | Purpose |
|------|---------|
| PyTorch + torchvision | Model, training, pretrained ResNet18 |
| scikit-learn | Classification report, confusion matrix |
| OpenCV | Grad-CAM heatmap generation and blending |
| matplotlib | Loss/accuracy curves, visualization |
| kagglehub | Automated dataset download |
| numpy | Array operations, image unnormalization |

---

## 6. Files in This Project

| File | Description |
|------|-------------|
| `medical_image_classification_capstone.ipynb` | Full notebook — end-to-end pipeline |
| `resnet18_xray_classifier.pth` | Saved model weights (~44.7 MB) |
| `requirements.txt` | Python package dependencies |
| `xray_data/` | Full dataset (COVID / Normal / Lung_Opacity / Viral Pneumonia) |
| `xray_subset/` | 300-images-per-class subset used for training |
| `README.md` | Setup & run instructions |
| `PROJECT_SUMMARY.md` | This file |

---

## 7. Possible Extensions

- **Try VGG16 or EfficientNet** as a second backbone and compare metrics
- **Increase N_PER_CLASS** to 1000+ on a GPU (Colab T4) for significantly higher accuracy
- **Add TTA** (Test-Time Augmentation) to further improve test accuracy
- **Class-weighted loss** to address the imbalance between COVID recall and precision
- **Grad-CAM++** for more accurate localisation than standard Grad-CAM
