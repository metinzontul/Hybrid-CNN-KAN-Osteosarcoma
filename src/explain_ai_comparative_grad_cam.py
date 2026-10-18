# -*- coding: utf-8 -*-


"""
Comparative Explainable AI (XAI) Pipeline using Grad-CAM

This script selects three distinct clinical scenarios from the dataset:
1. Malignant (Osteosarcoma)
2. Benign Tumor (Non-cancerous)
3. Healthy Bone (No tumor)

It generates Grad-CAM heatmaps for all three to demonstrate the model's 
discriminative focus and saves a combined publication-quality figure.

Author: Metin Zontul
Date: October 2026
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model(model_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Loads the EfficientNetV2-S architecture with trained weights."""
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path: str, img_size: int = 224) -> tuple:
    """Reads and preprocesses the image for both visualization and model input."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    
    rgb_img = np.float32(img) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0) 
    
    return rgb_img, input_tensor

def get_representative_images(excel_path: str) -> dict:
    """
    Scans the metadata to automatically select one image ID for each clinical scenario.
    """
    df = pd.read_excel(excel_path)
    
    # 1. Osteosarcoma (Malignant)
    osteo_img = df[df['osteosarcoma'] == 1]['image_id'].iloc[0]
    
    # 2. Benign Tumor (Non-osteosarcoma but has a tumor)
    benign_img = df[(df['benign'] == 1) & (df['osteosarcoma'] == 0)]['image_id'].iloc[0]
    
    # 3. Healthy (No tumor at all)
    healthy_img = df[(df['tumor'] == 0) & (df['osteosarcoma'] == 0)]['image_id'].iloc[0]
    
    return {
        "Osteosarcoma (Cancer)": osteo_img,
        "Benign Tumor (Non-Cancer)": benign_img,
        "Healthy Bone (No Tumor)": healthy_img
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Comparative Grad-CAM heatmaps.")
    parser.add_argument('--base_dir', type=str, default=r"C:\Users\ASUS-WORKSTATION\Desktop\metin\Akif\BTXRD")
    args = parser.parse_args()

    # Paths
    model_path = os.path.join(args.base_dir, "saved_models", "best_model.pth")
    excel_path = os.path.join(args.base_dir, "data", "raw", "metadata", "dataset.xlsx")
    raw_img_dir = os.path.join(args.base_dir, "data", "raw", "images")
    save_dir = os.path.join(args.base_dir, "logs", "grad_cam")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path, device)
    
    # Target the last convolutional block of EfficientNetV2
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)] # We strictly ask: "Look for Osteosarcoma features"

    # Fetch Images
    image_dict = get_representative_images(excel_path)
    
    # Setup Plot (3 rows, 2 columns) - 600 DPI for publication
    fig, axes = plt.subplots(3, 2, figsize=(10, 15), dpi=600)
    fig.suptitle("Grad-CAM Focus: Differentiating Clinical Scenarios", fontsize=18, y=0.98)

    for i, (condition, img_name) in enumerate(image_dict.items()):
        img_path = os.path.join(raw_img_dir, img_name)
        logging.info(f"Processing {condition}: {img_name}")
        
        rgb_img, input_tensor = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)

        # 1. Get Model's Prediction Probability
        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)[0, 1].item() * 100 # Osteosarcoma probability %

        # 2. Generate Grad-CAM Heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 3. Plotting
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"Original ({condition})", fontsize=14)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam_image)
        # Adding the prediction confidence to the title for clinical context
        axes[i, 1].set_title(f"Grad-CAM (Osteosarcoma Prob: {prob:.1f}%)", fontsize=14)
        axes[i, 1].axis('off')

    plt.tight_layout()
    
    # Save the combined figure
    save_path = os.path.join(save_dir, "Comparative_GradCAM.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparative Grad-CAM visualization saved successfully to {save_path}")

if __name__ == "__main__":
    main()