# 🧠 EEG Sleep Deprivation Classification

Deep learning pipeline for classifying **Normal Sleep (NS)** vs **Sleep Deprivation (SD)** using resting-state EEG data.

This project implements an EEGNet-based convolutional neural network trained on 2-second EEG epochs extracted from resting-state recordings.

---

# 📁 Project Structure

```
EEEG_Sleep_Deprivation/
│
├── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── data/
│   └── preprocessed/
│
├── saved_models/
├── main.py
└── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ethics-neuro-ai/EEEG_Sleep_Deprivation.git
cd EEEG_Sleep_Deprivation
```

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset Structure

The dataset must follow this structure:

```
dataset_path/
├── sub-01/
│   ├── ses-1/eeg/*.set
│   ├── ses-2/eeg/*.set
├── sub-02/
...
```

- `ses-1` → Normal Sleep (label = 0)
- `ses-2` → Sleep Deprivation (label = 1)
- File format: EEGLAB `.set`

---

# ▶️ Running the Full Pipeline

### Edit paths in `main.py`:

```python
DATA_PATH = "/path/to/raw_dataset"
SAVE_PATH = "./data/preprocessed"
MODEL_PATH = "./saved_models/eegnet_sleep_model.keras"
```

### Run:

```bash
python main.py
```

---

# 🔄 Pipeline Overview

## 1️⃣ Preprocessing
- Bandpass filter (1–45 Hz)
- Average re-reference
- Resample to 125 Hz
- 2-second fixed-length epochs
- Per-epoch normalization
- Save as `.npy` files

## 2️⃣ Training
- EEGNet-based CNN
- 70/15/15 train/validation/test split
- Binary classification (NS vs SD)

## 3️⃣ Evaluation
- Test Accuracy
- ROC Curve & AUC
- Confusion Matrix
- Precision-Recall Curve
- Sensitivity & Specificity

## 4️⃣ Model Saving
Saved to:

```
saved_models/eegnet_sleep_model.keras
```

---

# 🧠 Model Description

The model learns:

- Spatial patterns across electrodes
- Temporal oscillatory dynamics
- Physiological differences between normal sleep and sleep deprivation

**Input shape:**
```
(n_channels, n_times, 1)
```

**Output:**
```
Probability of Sleep Deprivation
```

---

# 📈 Example Performance

```
Test Accuracy: ~95%
ROC AUC: ~0.99
Balanced precision and recall across classes
```

---

# 🛠 System Requirements

- Python ≥ 3.9
- ≥ 8GB RAM recommended
- GPU optional but speeds up training

---

# ⚠️ Important Note

Current evaluation uses **epoch-level splitting**, meaning:

- Data from the same subject may appear in both train and test sets.
- Results reflect strong within-subject discriminability.
- Subject-wise validation is recommended for real-world generalization.

---

# 📌 Research Purpose

This project demonstrates the feasibility of using convolutional neural networks to detect physiological alterations in resting-state EEG caused by sleep deprivation.

It is intended for research and methodological exploration.

---
