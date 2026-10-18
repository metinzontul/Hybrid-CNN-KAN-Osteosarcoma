# Exploratory Evaluation of a Hybrid EfficientNetV2-S and Kolmogorov-Arnold Network for Osteosarcoma-versus-Rest Classification in Radiographs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-BTXRD-blue)](https://doi.org/10.6084/m9.figshare.27865398)

This repository contains the official PyTorch implementation of the paper: **"Exploratory Evaluation of a Hybrid EfficientNetV2-S and Kolmogorov-Arnold Network for Osteosarcoma-versus-Rest Classification in Radiographs"** (published in *MDPI Diagnostics*).

**Authors:** Uğur Özdemir, Metin Zontul, Seda Karateke, Mehmet Akif Cifci, Aykut Başer

## 📌 Overview

Automated radiographic classification of osteosarcoma is highly challenging due to morphological heterogeneity and extreme class imbalance. This project introduces a hybrid deep learning framework that replaces the traditional Multi-Layer Perceptron (MLP) classification head with a B-spline-based **Kolmogorov-Arnold Network (KAN)**. 

By integrating an ImageNet-pretrained **EfficientNetV2-S** backbone with a constrained KAN classifier, and utilizing **Focal Loss ($\gamma=2.0$)** with weighted random sampling, the proposed architecture provides a highly specific, exploratory "second-opinion" baseline for primary bone tumor classification.

## 📂 Repository Structure

```text
OsteoKAN/
├── dataset/                  # Directory for the BTXRD dataset (Not included in repo)
├── src/                      # Source code
│   ├── model.py              # Hybrid CNN-KAN architecture definition
│   ├── data_loader.py        # Dataloaders, augmentation, and WeightedRandomSampler
│   ├── train.py              # Training loop with Focal Loss and validation checkpointing
│   └── test_hybrid_kan.py    # Independent test evaluation and Confusion Matrix generation
├── notebooks/                # Jupyter Notebooks for visualization
│   └── xai_gradcam.ipynb     # Exploratory Grad-CAM saliency maps (Figures 4, 5, 6)
├── saved_models/             # Directory for saving best .pth weights
├── logs/                     # Directory for saving outputs (e.g., confusion_matrix.png)
├── .gitignore                # Ignored files to prevent data/weight leaks
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OsteoKAN.git
   cd OsteoKAN
   ```
2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   conda create -n osteokan python=3.10 -y
   conda activate osteokan
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation (BTXRD)

Due to GitHub's storage limits and data licensing, the dataset is **not** hosted in this repository. 

1. Download the original **Bone Tumor X-ray Radiograph Dataset (BTXRD)** from [Figshare](https://doi.org/10.6084/m9.figshare.27865398).
2. Extract the images and apply the stratified image-level split (70% Train, 15% Val, 15% Test) as described in the paper.
3. Organize the dataset strictly into the following directory structure inside the `dataset/` folder:

```text
dataset/
├── train/
│   ├── osteosarcoma/
│   └── non-osteosarcoma/
├── val/
│   ├── osteosarcoma/
│   └── non-osteosarcoma/
└── test/
    ├── osteosarcoma/
    └── non-osteosarcoma/
```

## 🚀 Usage

### 1. Training the Model
To train the Hybrid CNN-KAN model from scratch using the settings reported in the paper (Batch Size=32, LR=1e-4, Focal Loss $\gamma=2.0$), run:
```bash
cd src
python train.py
```
*The script will automatically save the champion model (based on Validation PR-AUC) into the `saved_models/` directory.*

### 2. Testing and Evaluation
To evaluate the trained model on the strictly isolated test set and generate the confusion matrix:
```bash
cd src
python test_hybrid_kan.py
```
*The resulting confusion matrix plot will be saved in the `logs/` directory.*

## 📈 Results (Multi-Seed Evaluation)

The hybrid configuration was evaluated across three random training seeds (42, 1234, 2026) on the held-out image-level test set ($N=562$). The table below summarizes the run-to-run variability (Mean $\pm$ SD):

| Metric | Multi-Seed Performance | Interpretation |
| :--- | :--- | :--- |
| **PR-AUC** | 0.9053 ± 0.0122 | Primary evaluation metric |
| **Specificity** | 0.9936 ± 0.0018 | High precision, low false-positive rate |
| **Sensitivity** | 0.7652 ± 0.0283 | Detection of osteosarcoma-positive images |
| **F1-Score** | 0.8312 ± 0.0262 | Balance between precision and sensitivity |

## 📝 Citation
If you find this code or research helpful in your work, please cite our paper:

```bibtex
@article{ozdemir2026exploratory,
  title={Exploratory Evaluation of a Hybrid EfficientNetV2-S and Kolmogorov-Arnold Network for Osteosarcoma-versus-Rest Classification in Radiographs},
  author={Özdemir, Uğur and Zontul, Metin and Karateke, Seda and Cifci, Mehmet Akif and Başer, Aykut},
  journal={Diagnostics},
  year={2026},
  publisher={MDPI}
}
```
*(Update volume/issue/pages once assigned by MDPI).*