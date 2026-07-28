"""Training visualization utilities"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from rich.text import Text
from rich.padding import Padding
from typing import Union
from .utils import save_plot
from ..richlog import StepHandle, Logger, NULL_STEP
from ..richlog.core import INDENT


def plot_training_history(history: Dict[str, List], training_time: float,
                          experiment_name: Optional[str] = None, save_dir: str = 'results',
                          run_timestamp: Optional[str] = None, *, step: StepHandle = NULL_STEP) -> None:
    """Plot training history and save as PNG file"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    epochs = np.array(history['epoch'])
    train_losses = np.array(history['train_loss'])
    val_losses = np.array(history['val_loss'])
    
    train_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100)
    val_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0] * 100)
    
    fig.suptitle(
        f'Training History | Time: {training_time:.2f}s | Train Δ: {train_improvement:.1f}% | Val Δ: {val_improvement:.1f}%', 
        fontsize=13, fontweight='bold'
    )
    
    ax.plot(epochs, train_losses, marker='o', linewidth=2.5, markersize=7, 
            color='#2E86AB', label='Train Loss', markerfacecolor='white', markeredgewidth=2)
    ax.plot(epochs, val_losses, marker='s', linewidth=2.5, markersize=7, 
            color='#A23B72', label='Val Loss', markerfacecolor='white', markeredgewidth=2)
    
    ax.annotate(f'{train_losses[-1]:.4f}', xy=(epochs[-1], train_losses[-1]), 
                xytext=(10, 0), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))
    ax.annotate(f'{val_losses[-1]:.4f}', xy=(epochs[-1], val_losses[-1]), 
                xytext=(10, 0), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightpink', alpha=0.7))
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Train vs Validation Loss', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save plot instead of showing it
    if experiment_name:
        save_plot(fig, experiment_name, f'training_history_{int(training_time)}s.png', save_dir, run_timestamp, step=step)
    else:
        plt.show()
    
    plt.close(fig)


def print_metrics_summary(metrics_dict: Dict[str, Dict[str, float]],
                          *, step: Union[StepHandle, Logger] = NULL_STEP) -> None:
    """Log a metrics table as one permanent block (.block()), so it
    survives in the terminal even after the step it was computed under has
    already closed and collapsed.

    Accepts either a StepHandle (the usual mid-run case) or the bare Logger
    (e.g. main.py's final summary, printed after every step has closed) —
    both expose the same .block() method.
    """
    metrics_dict = {k.lower(): v for k, v in metrics_dict.items()}
    dataset_splits = list(metrics_dict.keys())
    metrics = next(iter(metrics_dict.values()))

    lines = Text()
    for i, metric in enumerate(metrics):
        parts = []
        for split in dataset_splits:
            value = metrics_dict[split][metric]
            parts.append(f'{split}={value:.4f}')

        if i > 0:
            lines.append("\n")
        lines.append(f'  {metric + ":":<12}', style="bold")
        lines.append(" | ".join(parts), style="dim")

    padded_lines = Padding(lines, (0, 0, 0, len(INDENT) * 3))
    step.block(padded_lines)
