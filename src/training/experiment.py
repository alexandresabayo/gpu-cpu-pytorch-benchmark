"""Experiment running and comparison utilities"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from ..training.core import train_model
from ..utils.metrics import calculate_metrics
from ..visualization.training import plot_training_history, print_metrics_summary
from ..richlog import StepHandle, NULL_STEP


def run_training_experiment(model: nn.Module, loaders: Dict[str, DataLoader],
                            criterion, device: torch.device, config, experiment_name: Optional[str] = None,
                            run_timestamp: Optional[str] = None, *, step: StepHandle = NULL_STEP) -> Tuple[float, Dict]:
    """Run complete training experiment on a single device.

    Opens a "Training on CPU"/"Training on GPU" child step under `step` for
    the training loop itself. That child step closes (and collapses to one
    permanent line) before the metrics table is printed, so the metrics
    table survives as its own permanent block rather than getting collapsed
    away with the training detail.
    """
    device_label = "Training on CPU" if device.type == "cpu" else "Training on GPU"

    with step.child(device_label) as dstep:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Ensure CUDA kernels are compiled before timing measurements
        if device.type == 'cuda':
            torch.cuda.synchronize()

        history, training_time = train_model(model, loaders, criterion, optimizer, device, config, step=dstep)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metrics = {
            'train': calculate_metrics(model, loaders['train'], criterion, device),
            'val': calculate_metrics(model, loaders['val'], criterion, device),
            'test': calculate_metrics(model, loaders['test'], criterion, device)
        }

        if config.training_history:
            plot_training_history(history, training_time, experiment_name,
                                  save_dir='results', run_timestamp=run_timestamp, step=dstep)

        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None

    # dstep has now closed and collapsed to its one permanent line. The
    # header + metrics table below are printed as their own permanent
    # blocks, so they stay visible regardless of that collapse.
    header = f"{device.type.upper()} training time: {training_time:.2f}s"
    if gpu_mem is not None:
        header += f" | GPU memory used: {gpu_mem:.2f} GB"
    step.block(header, indent=3)
    print_metrics_summary(metrics, step=step)

    return training_time, metrics


def run_experiment(experiment_name: str, model_cpu: nn.Module, model_gpu: nn.Module,
                   loaders: Dict[str, DataLoader], criterion, config, run_timestamp: Optional[str] = None,
                   *, step: StepHandle = NULL_STEP) -> dict:
    """Run complete CPU vs GPU comparison experiment.

    `step` is expected to already be open (e.g. main.py's per-experiment
    step, titled with the experiment name) — this function logs into it and
    opens the CPU/GPU child steps, but does not open the experiment-level
    step itself.
    """
    n_params = sum(p.numel() for p in model_cpu.parameters())
    step.block(f"model parameters: {n_params:,}", indent=3, style="dim")

    cpu_time, cpu_metrics = run_training_experiment(
        model_cpu, loaders, criterion, torch.device('cpu'), config, experiment_name, run_timestamp, step=step
    )

    gpu_time = None
    gpu_metrics = None
    if torch.cuda.is_available():
        gpu_time, gpu_metrics = run_training_experiment(
            model_gpu, loaders, criterion, torch.device('cuda'), config, experiment_name, run_timestamp, step=step
        )

        speedup = cpu_time / gpu_time
        step.block(f"speedup: {speedup:.2f}x", indent=3, style="dim")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        step.warn("GPU not available")

    return {
        'name': experiment_name,
        'n_params': n_params,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'cpu_metrics': cpu_metrics,
        'gpu_metrics': gpu_metrics
    }
