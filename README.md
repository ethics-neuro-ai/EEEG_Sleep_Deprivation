# EEGNet Epoch-Level Training

This repository contains code for training **EEGNet** on the **Sleep Deprivation EEG dataset** at the **epoch level**.

## Dataset

- **Name:** Sleep Deprivation EEG (OpenNeuro: [ds004902](https://openneuro.org/datasets/ds004902))  
- **Subjects:** 71 participants  
- **Sessions:** Normal Sleep (NS, session 1) and Sleep Deprivation (SD, session 2)  
- **Data:** Resting-state EEG (eyes open/closed), with behavioral measures of sleepiness and mood  
- **Format:** EEGLAB `.set` and `.fdt` files  

> **Note:** The dataset files are not included in this repository. Download them separately and place in the `data/` folder.




## Features

- **Epoch-level training:** Loads all epochs from all subjects into memory at once  
- **EEGNet model:** Standard deep learning architecture for EEG classification  
- **Metrics:** Accuracy, loss, ROC AUC, confusion matrix, classification report  

## Usage

1. Clone this repository:


git clone https://github.com/ethics-neuro-ai/eeg-epoch-level.git
cd eeg-epoch-level


Download the dataset from OpenNeuro and place all .set/.fdt files in the data/ folder.

Open the Jupyter notebook:

jupyter notebook notebooks/epoch_level_training.ipynb

Run the cells to preprocess data, train EEGNet, and evaluate performance.

Notes

Epoch-level vs Subject-wise:

Epoch-level: all epochs from all subjects loaded into RAM (requires sufficient memory, e.g., 8GB+)

Subject-wise: streams data per subject to reduce RAM usage, suitable for real-time or low-memory scenarios (implemented in a separate notebook).

Data Exclusion: Files with missing or corrupt samples are skipped automatically.

Results: High classification performance on the sleep deprivation EEG dataset, with ROC-AUC ~0.99 and accuracy ~0.96.
