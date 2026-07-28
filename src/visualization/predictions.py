"""Prediction visualization utilities"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Optional
from .utils import save_plot
from ..richlog import StepHandle, NULL_STEP


def visualize_predictions(experiment_name: str, model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          num_samples: int = 3, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          save_dir: str = 'results', run_timestamp: Optional[str] = None,
                          *, step: StepHandle = NULL_STEP) -> Tuple[np.ndarray, np.ndarray]:
    """Visualize model predictions and save as PNG files"""
    model.eval()
    model.to(device)

    inputs, targets = next(iter(data_loader))
    inputs = inputs[:num_samples].to(device)
    targets = targets[:num_samples].to(device)

    with torch.no_grad():
        predictions = model(inputs)

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Helper to remove trailing singleton dimensions
    def squeeze_last(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 1:
            return x.reshape(x.shape[0], x.shape[1])
        return x

    inputs = squeeze_last(inputs)
    targets = squeeze_last(targets)
    predictions = squeeze_last(predictions)

    # Ensure consistent dimensionality for single-sample cases
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    output_size = predictions.shape[1]
    is_classification = targets.dtype in [np.int32, np.int64] and output_size >= 2
    
    if is_classification:
        fig = _plot_mnist_predictions(experiment_name, inputs, predictions, targets, num_samples)
    else:
        fig = _plot_timeseries_predictions(experiment_name, inputs, predictions, targets, num_samples)
    
    # Save the plot
    if fig is not None:
        save_plot(fig, experiment_name, 'predictions.png', save_dir, run_timestamp, step=step)
        plt.close(fig)
    
    return predictions, targets


def _plot_mnist_predictions(experiment_name: str, inputs: np.ndarray, predictions: np.ndarray, 
                            targets: np.ndarray, num_samples: int):
    """Helper: plot MNIST predictions and return figure"""
    fig, axes = plt.subplots(1, num_samples, figsize=(8, 3))
    fig.suptitle(f'{experiment_name} - Model Predictions vs Actual',
                 fontsize=16, fontweight='bold')

    for idx in range(num_samples):
        ax = axes[idx]
        img = inputs[idx].reshape(28, 28)
        pred_label = np.argmax(predictions[idx])
        actual_label = targets.squeeze()[idx]

        ax.imshow(img, cmap='gray')
   # Visual feedback: green for correct, red for incorrect predictions
        color = 'green' if pred_label == actual_label else 'red'
        ax.set_title(f'Pred: {pred_label}\nActual: {actual_label}',
                     fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    plt.tight_layout()
    return fig


def _plot_timeseries_predictions(experiment_name: str, inputs: np.ndarray, predictions: np.ndarray,
                                 targets: np.ndarray, num_samples: int):
    """Helper: plot time series predictions and return figure"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    fig.suptitle(f'{experiment_name} - Model Predictions vs Actual', 
                 fontsize=16, fontweight='bold')

    for idx in range(num_samples):
        ax = axes[idx]
        input_len = inputs.shape[1]
        pred_len = predictions.shape[1]

        # Separate time axes for historical inputs and future predictions
        input_times = np.arange(0, input_len)
        future_times = np.arange(input_len, input_len + pred_len)

        ax.plot(input_times, inputs[idx], label='History (inputs)',
                linewidth=2, marker='o', markersize=4, alpha=0.9, color='grey')
        ax.plot(future_times, targets[idx], label='Target / Future (actual)',
                linewidth=2.5, marker='s', markersize=5, alpha=0.95, color='black')
        ax.plot(future_times, predictions[idx], label='Predicted (model)',
                linewidth=2.5, linestyle='--', marker='D', markersize=5, 
                alpha=0.8, color='tab:orange')
        ax.axvline(x=input_len - 0.5, color='red', linestyle=':', 
                   linewidth=2, alpha=0.6, label='Forecast start')

        mse = np.mean((targets[idx] - predictions[idx]) ** 2)
        mae = np.mean(np.abs(targets[idx] - predictions[idx]))

        # Plot formatting
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'Sample {idx + 1} | Forecast MSE: {mse:.4f} | MAE: {mae:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    return fig