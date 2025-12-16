"""Core training and evaluation functions"""

import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
from ..utils.config import TrainingConfig

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


def train_step_with_progress(model: nn.Module, data_loader: DataLoader, 
                             criterion, optimizer, device: torch.device) -> float:
    """Training logic for one epoch with progress bar (returns average loss)."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    train_pbar = tqdm(data_loader, ascii=True, leave=False, ncols=63, file=sys.stdout,
                      bar_format='{n_fmt}/{total_fmt} [{bar}] {remaining}{postfix}')
    
    for inputs, targets in train_pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_count += 1
        
        train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    train_loss = total_loss / batch_count
    train_pbar.set_postfix({'loss': f'{train_loss:.4f}'})
    print(str(train_pbar), end='')
    
    return train_loss


def eval_step_with_progress(model: nn.Module, data_loader: DataLoader, 
                            criterion, device: torch.device) -> float:
    """Validation logic for one epoch with progress bar (returns average loss)."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1
    
    val_loss = total_loss / batch_count
    print(f', val_loss={val_loss:.4f}')
    
    return val_loss


def train_model(model: nn.Module, loaders: Dict[str, DataLoader], 
                criterion, optimizer, device: torch.device, 
                config: TrainingConfig) -> Tuple[Dict[str, List], float]:
    """Core training loop - returns history and time"""
    model.to(device)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Choose training functions based on config
    train_step_fn = train_step_with_progress if config.show_progress else train_step
    eval_step_fn = eval_step_with_progress if config.show_progress else eval_step
    
    # Setup epoch-level progress bar if early stopping is enabled without batch progress
    use_epoch_pbar = config.early_stopping and not config.show_progress
    if use_epoch_pbar:
        bar_format = 'Epoch {n_fmt}/{total_fmt} |{bar}| {remaining}{postfix}'
        epoch_pbar = tqdm(range(config.epochs), bar_format=bar_format, ncols=80)
    else:
        epoch_pbar = range(config.epochs)
    
    start_time = time.perf_counter()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in epoch_pbar:
        if config.show_progress:
            print(f'Epoch {epoch + 1}/{config.epochs}')
        
        train_loss = train_step_fn(model, loaders['train'], criterion, optimizer, device)
        val_loss = eval_step_fn(model, loaders['val'], criterion, device)
        
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
        
        # Update epoch progress bar with loss info (AFTER patience update)
        if use_epoch_pbar:
            postfix_dict = {
                'loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            }
            if config.early_stopping:
                postfix_dict['patience'] = f'{patience_counter}/{config.patience}'
            epoch_pbar.set_postfix(postfix_dict) # type: ignore
        
        # Check early stopping AFTER updating display
        if config.early_stopping and patience_counter >= config.patience:
            if use_epoch_pbar:
                epoch_pbar.close() # type: ignore
            print(f"Early stopping triggered.")
            break
    
    # Close progress bar if it exists
    if use_epoch_pbar and hasattr(epoch_pbar, 'close'):
        epoch_pbar.close() # type: ignore
    
    # Restore best model if early stopping was used
    if config.early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = time.perf_counter() - start_time
    
    return history, training_time