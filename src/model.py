# -*- coding: utf-8 -*-
"""
Hybrid EfficientNetV2-S and Kolmogorov-Arnold Network (KAN) Architecture
Exploratory Evaluation for Osteosarcoma-versus-Rest Classification

Author: Metin Zontul
Date: October 2026
"""

import torch
import torch.nn as nn
from torchvision import models

# Note: Ensure that the 'efficient_kan' package or PyKAN is accessible in your environment.
from efficient_kan import KAN 

class HybridCNN_KAN(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridCNN_KAN, self).__init__()
        
        # Load ImageNet-pretrained EfficientNetV2-S backbone
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_v2_s(weights=weights)
        
        # Extract the number of input features to the original classifier (1280)
        num_ftrs = self.backbone.classifier[1].in_features
        
        # Remove the original MLP classification head
        self.backbone.classifier[1] = nn.Identity()
        
        # Initialize the Spline-based KAN Classifier
        # Configuration based on the ablation study: Grid Size = 5, Spline Order = 3, Hidden Nodes = 32
        self.kan_classifier = KAN(
            [num_ftrs, 32, num_classes], 
            grid_size=5, 
            spline_order=3
        )

    def forward(self, x):
        # Extract 1D feature vector from the backbone
        features = self.backbone(x)
        
        # Pass features through the KAN classifier
        output = self.kan_classifier(features)
        
        return output

if __name__ == "__main__":
    # Test model initialization
    model = HybridCNN_KAN(num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [1, 2])")