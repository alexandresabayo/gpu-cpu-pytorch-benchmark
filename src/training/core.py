"""Core training and evaluation functions"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from ..utils.config import TrainingConfig
from ..richlog import StepHandle


def train_step(model: nn.Module, data_loader: DataLoader, criterion, 
               optimizer, device: torch.device) -> float:
    """Training logic for one epoch (returns average loss)"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / batch_count


def eval_step(model: nn.Module, data_loader: DataLoader, 
              criterion, device: torch.device) -> float:  
    """Validation logic for one epoch (returns average loss)"""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count


def train_model(step: StepHandle, model: nn.Module, loaders: Dict[str, DataLoader], 
                criterion, optimizer, device: torch.device, 
                config: TrainingConfig) -> Tuple[Dict[str, List], float]:
    """Core training loop - returns history and time.

    Progress is always shown as one epoch-level progress bar (regardless of
    early_stopping), with a rolling window of per-epoch summary lines
    underneath it. There is no separate per-batch verbosity mode: batch-level
    detail was dropped entirely, so this is the single training loop rather
    than a choice between two.
    """
    model.to(device)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    prog = step.progress(total=config.epochs, label="epoch")

    start_time = time.perf_counter()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config.epochs):
        train_loss = train_step(model, loaders['train'], criterion, optimizer, device)
        val_loss = eval_step(model, loaders['val'], criterion, device)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping logic
        if config.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

        summary = f"epoch {epoch + 1}/{config.epochs}: loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if config.early_stopping:
            summary += f" patience={patience_counter}/{config.patience}"
        step.info(summary)
        prog.advance()

        # Check early stopping AFTER logging the epoch's own line
        if config.early_stopping and patience_counter >= config.patience:
            step.warn(f"early stopping triggered at epoch {epoch + 1}")
            break
    
    # Restore best model if early stopping was used
    if config.early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = time.perf_counter() - start_time
    
    return history, training_time
