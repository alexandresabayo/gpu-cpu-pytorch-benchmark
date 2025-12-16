"""Experiment running and comparison utilities"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from ..training.core import train_model
from ..utils.metrics import calculate_metrics
from ..visualization.training import plot_training_history, print_metrics_summary



def run_training_experiment(model: nn.Module, loaders: Dict[str, DataLoader],
                            criterion, device: torch.device, config, experiment_name: Optional[str] = None,
                            run_timestamp: Optional[str] = None) -> Tuple[float, Dict]:
    """Run complete training experiment on a single device"""
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Ensure CUDA kernels are compiled before timing measurements
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    history, training_time = train_model(model, loaders, criterion, optimizer, device, config)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    metrics = {
        'train': calculate_metrics(model, loaders['train'], criterion, device),
        'val': calculate_metrics(model, loaders['val'], criterion, device),
        'test': calculate_metrics(model, loaders['test'], criterion, device)
    }
    
    if config.training_history:
        plot_training_history(history, training_time, experiment_name, save_dir='results', run_timestamp=run_timestamp)
    
    print()
    print_metrics_summary(metrics)
    print(f'\n{device.type.upper()} Training Time: {training_time:.2f} seconds')
    
    if device.type == 'cuda':
        print(f'GPU Memory Used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    
    print()
    
    return training_time, metrics


def run_experiment(experiment_name: str, model_cpu: nn.Module, model_gpu: nn.Module,
                   loaders: Dict[str, DataLoader], criterion, config, run_timestamp: Optional[str] = None) -> dict:
    """Run complete CPU vs GPU comparison experiment"""
    print(f'\n{"="*80}')
    print(f'EXPERIMENT: {experiment_name}')
    print(f'{"="*80}')
    
    n_params = sum(p.numel() for p in model_cpu.parameters())
    print(f'Model Parameters: {n_params:,}\n')
    
    print(f'\n--- Training on CPU ---\n')
    cpu_time, cpu_metrics = run_training_experiment(
        model_cpu, loaders, criterion, torch.device('cpu'), config, experiment_name, run_timestamp
    )
    
    gpu_time = None
    gpu_metrics = None
    if torch.cuda.is_available():
        print(f'\n--- Training on GPU ---\n')
        gpu_time, gpu_metrics = run_training_experiment(
            model_gpu, loaders, criterion, torch.device('cuda'), config, experiment_name, run_timestamp
        )
        
        speedup = cpu_time / gpu_time
        print(f'\nSpeedup: {speedup:.2f}x')
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        print('\nGPU not available!')
    
    return {
        'name': experiment_name,
        'n_params': n_params,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'cpu_metrics': cpu_metrics,
        'gpu_metrics': gpu_metrics
    }