# -*- coding: utf-8 -*-
"""
Training Loop with Focal Loss and Checkpointing based on Validation PR-AUC.

Author: Metin Zontul
Date: October 2026
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score

from model import HybridCNN_KAN
from data_loader import get_dataloaders

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def train_model(data_dir, save_dir, num_epochs=30, seed=42):
    torch.manual_seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(data_dir, batch_size=32)
    
    model = HybridCNN_KAN(num_classes=2).to(device)
    
    # Focal Loss (gamma=2.0) and AdamW with shared learning rate (1e-4)
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    best_pr_auc = 0.0
    
    print(f"Starting training on {device} with Seed {seed}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1] 
                
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_pr_auc = average_precision_score(val_labels, val_probs)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(dataloaders['train'].dataset):.4f} - Val PR-AUC: {val_pr_auc:.4f}")
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            save_path = os.path.join(save_dir, f'best_model_seed_{seed}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new champion model with PR-AUC: {best_pr_auc:.4f}")

if __name__ == "__main__":
    # Dinamik dosya yolları (Projenin ana dizinini bulur)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Çoklu tohumlu döngü için seed değerini değiştirebilirsiniz (42, 1234, 2026)
    train_model(DATA_DIR, SAVE_DIR, num_epochs=30, seed=42)