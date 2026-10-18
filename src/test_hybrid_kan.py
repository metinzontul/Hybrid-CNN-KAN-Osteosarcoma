# -*- coding: utf-8 -*-
"""
Test Pipeline and Confusion Matrix Generation for Hybrid CNN-KAN Model
Author: Metin Zontul
Date: October 2026
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
import logging

from model import HybridCNN_KAN
from data_loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    logging.info("Starting evaluation on the independent Test Set...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, save_path):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fig.patch.set_facecolor('white')
    
    # Makaledeki ayrışmayı sağlamak için Yeşil tonları (Greens) kullanıldı
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-Osteosarcoma (0)', 'Osteosarcoma (1)'],
                yticklabels=['Non-Osteosarcoma (0)', 'Osteosarcoma (1)'],
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)

    ax.set_xlabel('AI Model Prediction', fontsize=14, labelpad=10)
    ax.set_ylabel('True Clinical Diagnosis', fontsize=14, labelpad=10)
    ax.set_title('Test Set Confusion Matrix (Hybrid CNN-KAN)', fontsize=16, pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Confusion matrix saved to: {save_path}")

def main():
    # Dinamik dosya yolları
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_model_seed_42.pth")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloaders = get_dataloaders(DATA_DIR, batch_size=32)
    test_loader = dataloaders['test']
    
    model = HybridCNN_KAN(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    labels, preds, probs = evaluate_on_test_set(model, test_loader, device)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pr_auc = average_precision_score(labels, probs)
    f1 = f1_score(labels, preds)
    
    logging.info(f"--- FINAL TEST SET RESULTS ---")
    logging.info(f"PR-AUC:      {pr_auc:.4f}")
    logging.info(f"Sensitivity: {sensitivity:.4f} ({tp}/{tp+fn})")
    logging.info(f"Specificity: {specificity:.4f} ({tn}/{tn+fp})")
    logging.info(f"F1-Score:    {f1:.4f}")
    
    cm_save_path = os.path.join(LOGS_DIR, "confusion_matrix_test_hybrid_kan.png")
    plot_confusion_matrix(cm, cm_save_path)

if __name__ == "__main__":
    main()