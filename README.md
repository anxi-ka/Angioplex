# ANGIOPLEX 
### Angiography Analysis and Stenosis Decision Support Platform

ANGIOPLEX is a deep learning-based clinical decision support tool that analyzes coronary angiography images to predict stenosis severity and recommend treatment strategies. It features two trained models an EfficientNet CNN and a custom CNN augmented with Convolutional Block Attention Modules (CBAM) presented through an interactive Streamlit dashboard.

---

## Features

- **Dual-model inference** — EfficientNet CNN (regression) and CNN+CBAM+Regression (dual-output)
- **Stenosis percentage prediction** from uploaded angiography images
- **Severity classification** across five levels: Minimal, Mild, Moderate, Severe, Critical
- **Treatment recommendation engine** — suggests intervention type, medication intensity, urgency level, and follow-up schedule
- **Exploratory Data Analysis (EDA) tab** — visualizes dataset distributions and statistics
- **Model comparison tab** — side-by-side architecture overview and training performance graphs

---

## Project Structure

```
Angioplex/
├── angioplex.py                  # Main Streamlit dashboard
├── test_02.ipynb                 # Training notebook (EDA, model training, evaluation)
├── requirements.txt              # Python dependencies
├── model_links.txt               # Google Drive links for .h5 model weight files
├── graph_statistics.png          # Dataset statistics visualization
└── graphs/                       # Training performance plots (accuracy, loss, MAE)
```

---

## Model Weights

The trained model weight files exceed GitHub's 25 MB file size limit and are hosted on Google Drive. Download both files and place them in the **same folder as `angioplex.py`** before running the app.

See **[model_links.txt](./model_links.txt)** for the download links.

| File | Size | Description |
|------|------|-------------|
| `efficientnet_stenosis_model.h5` | ~16 MB | EfficientNetB0 fine-tuned for stenosis regression |
| `hybrid_cnn_cbam_stenosis_model_lambda_free.h5` | ~62 MB | Custom CNN + CBAM dual-output model |

---

## Architecture Overview

### Model 1 — EfficientNet CNN
- Base: EfficientNetB0 pre-trained on ImageNet
- Head: Single regression output → stenosis percentage (0–100%)
- Input: 224×224×3 grayscale angiography image
- Parameters: ~5.3M

### Model 2 — CNN + CBAM + Regression
- Base: Custom 4-block CNN (64→128→256→512 filters)
- Attention: CBAM modules (channel + spatial) after each block
- Head: Dual output — regression (stenosis %) + classification (severity class)
- Input: 224×224×3 grayscale angiography image

### Severity Thresholds

| Severity | Stenosis Range |
|----------|---------------|
| 🟢 Minimal | < 30% |
| 🔵 Mild | 30–50% |
| 🟡 Moderate | 50–70% |
| 🟠 Severe | 70–85% |
| 🔴 Critical | > 85% |

---

## Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/anxi-ka/Angioplex.git
cd Angioplex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights
Download both `.h5` files from the links in `model_links.txt` and place them in the `Angioplex/` folder.

### 4. Run the dashboard
```bash
streamlit run angioplex.py
```

---

## Dependencies

```
streamlit
tensorflow
keras
numpy
pandas
matplotlib
pillow
```

---

## Dataset

The models were trained on a coronary angiography dataset with stenosis percentage labels and severity annotations across four classes. The training pipeline, augmentation strategy, and evaluation metrics are documented in `test_02.ipynb`.

---

## Disclaimer

> **This tool is intended for academic and research purposes only.** It is not a substitute for professional medical diagnosis or clinical judgment. Always consult a qualified healthcare professional for medical decisions.

---

## Author

**Anshika**
AI and IOT Automation Research Lab | Deep Learning & Medical Imaging
