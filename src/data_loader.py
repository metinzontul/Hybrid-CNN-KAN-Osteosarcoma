# -*- coding: utf-8 -*-
"""
Data Loader and Augmentation Pipeline for BTXRD Dataset
Handles ImageNet normalization and WeightedRandomSampler for high class imbalance.

Author: Metin Zontul
Date: October 2026
"""

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transforms with stochastic augmentations (Rotation up to 15 deg, flips)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation and Test transforms (No stochastic augmentations)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), eval_transforms),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), eval_transforms)
    }

    # Handling Class Imbalance with WeightedRandomSampler for the training set
    train_labels = [label for _, label in image_datasets['train'].samples]
    class_sample_counts = np.bincount(train_labels)
    weights = 1. / class_sample_counts
    samples_weights = weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=samples_weights, 
        num_samples=len(samples_weights), 
        replacement=True
    )

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    return dataloaders