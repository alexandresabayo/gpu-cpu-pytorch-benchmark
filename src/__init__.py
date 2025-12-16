"""
PyTorch GPU vs CPU Training Comparison Package

This package provides tools for comparing training performance between CPU and GPU
for various neural network architectures on different datasets.
"""

from .models import (
    LinearBased,
    Conv1dBased,
    Conv2dBased,
    LSTMBased
)
from .data import (
    load_as_tensor,
    normalize_inputs,
    prepare_labels,
    split_data,
    prepare_dataloaders
    )
from .training import (
    train_step,
    eval_step,
    train_model,
    run_training_experiment,
    run_experiment
)
from .utils import (
    TrainingConfig,
    TemperatureConfig,
    MNISTConfig,
    calculate_metrics,
    predict_batch
)
from .visualization import (
    plot_training_history,
    print_metrics_summary,
    visualize_predictions
)