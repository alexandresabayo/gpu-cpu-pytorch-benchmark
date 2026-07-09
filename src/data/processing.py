"""Data processing and preparation utilities"""

import torch
from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from ..utils.config import TrainingConfig


def normalize_inputs(X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalize inputs to [0,1] and return normalization params"""
    X_min = X.min().item()
    X_max = X.max().item()
    X_normalized = (X - X_min) / (X_max - X_min)
    params = {'min': X_min, 'max': X_max}
    return X_normalized, params


def prepare_labels(y: torch.Tensor, is_classification: bool) -> torch.Tensor:
    """For regression: normalize targets. For classification: convert to class indices"""
    if is_classification:
        return y.long().squeeze()
    else:
        y_min = y.min().item()
        y_max = y.max().item()
        return (y - y_min) / (y_max - y_min)


def split_data(X: torch.Tensor, y: torch.Tensor, 
               ratios: Tuple[float, float, float]) -> Tuple[Tuple, Tuple, Tuple]:
    """Split data into train/val/test"""
    train_ratio, val_ratio, _ = ratios
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_data = (X[:n_train], y[:n_train])
    val_data = (X[n_train:n_train+n_val], y[n_train:n_train+n_val])
    test_data = (X[n_train+n_val:], y[n_train+n_val:])
    
    return train_data, val_data, test_data


def prepare_dataloaders(X: torch.Tensor, y: torch.Tensor,
                        config: TrainingConfig) -> Dict[str, DataLoader]:
    """Prepare DataLoaders from NumPy arrays"""
    
    # Normalize to [0,1] for stable gradient flow
    X_norm, _ = normalize_inputs(X)
    
    # For regression: normalize targets. For classification: convert to class indices
    is_classification = y.shape[1] == 1 and torch.unique(y).numel() <= 10
    y_prep = prepare_labels(y, is_classification)
    
    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    train_data, val_data, test_data = split_data(X_norm, y_prep, ratios)
    
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    test_dataset = TensorDataset(*test_data)
    
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, config.batch_size, shuffle=True)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}