"""Metrics calculation utilities"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def predict_batch(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute all predictions and targets"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    return torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0)


def calculate_regression_metrics(predictions: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, float]:
    """Calculate regression metrics"""
    mae = torch.nn.L1Loss()(predictions, targets).item()
    mse = torch.nn.MSELoss()(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    ss_res = ((targets - predictions) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def calculate_classification_metrics(predictions: torch.Tensor,
                                     targets: torch.Tensor) -> Dict[str, float]:
    """Calculate classification metrics using sklearn."""
    # Convert to numpy and get predicted classes
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    pred_probs = torch.softmax(predictions, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    num_classes = predictions.shape[1]
    
    metrics = {}
    
    if num_classes == 2:
        metrics['f1'] = f1_score(targets_np, pred_classes)
        metrics['auc_roc'] = roc_auc_score(targets_np, pred_probs[:, 1])
    else:
        metrics['f1_macro'] = f1_score(targets_np, pred_classes, average='macro')
        metrics['auc_roc'] = roc_auc_score(targets_np, pred_probs,
                                           multi_class='ovr',average='macro')
        
    metrics['accuracy'] = accuracy_score(targets_np, pred_classes)
    
    return metrics


def calculate_metrics(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                      criterion, device: torch.device) -> Dict[str, float]:
    """Evaluate model and return metrics"""
    predictions, targets = predict_batch(model, data_loader, device)
    
    total_loss = criterion(predictions, targets.to(predictions.device)).item()
    metrics = {'loss': total_loss}
    
    # Determine if classification or regression
    is_classification = targets.dtype in [torch.long, torch.int, torch.int32, torch.int64]
    
    if is_classification:
        metrics.update(calculate_classification_metrics(predictions, targets))
    else:
        metrics.update(calculate_regression_metrics(predictions, targets))
    
    return metrics