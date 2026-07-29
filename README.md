## PyTorch Testbed

A modular pipeline for training and comparing neural network architectures (multi-layer perceptron, convolutional network, and recurrent network) on a time-series regression task (temperature) and an image classification task (MNIST digits). Each experiment trains the same architecture on CPU and GPU so you can compare training time and resulting model quality side by side.

### Features

- **Multiple model architectures**: MLP, 1D CNN, 2D CNN, LSTM
- **Multiple datasets**: Temperature time series, MNIST digits
- **CPU vs GPU comparison**: same architecture and data trained on both devices
- **Detailed metrics**: Training time, memory usage, accuracy, loss, etc.
- **Visualization**: Training history plots, prediction visualizations
- **Modular design**: Clean separation of concerns

### Requirements

* **conda** ([download Miniforge3](https://github.com/conda-forge/miniforge)) to create and manage the Python environment.

### Quickstart

```sh
make install    # Create the Conda environment and install dependencies
make data       # Generate/download the datasets
make run        # Run experiments
```

You can modify the `main.py` script to run only specific experiments by commenting out the ones you don't need.

### Experiments

The project runs 6 experiments comparing different model architectures on different datasets:

1. **Temperature - MLP**: Multi-Layer Perceptron on temperature time series data
2. **Temperature - CNN**: 1D CNN on temperature time series data  
3. **Temperature - LSTM**: LSTM on temperature time series data
4. **MNIST - MLP**: Multi-Layer Perceptron on MNIST digit classification
5. **MNIST - CNN**: 2D CNN on MNIST digit classification
6. **MNIST - LSTM**: LSTM on MNIST digit classification

### YAML configuration

Training parameters can be configured in `config/config.yaml`.  
You can also configure parameters in the `TrainingConfig` class.

- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `train_ratio`, `val_ratio`, `test_ratio`: Data split ratios
- `dropout`: Dropout rate
- `training_history`: Whether to plot training history
- `early_stopping`: Whether to use early stopping
- `patience`: Patience for early stopping

**Note**: YAML configuration takes precedence over default values when `config/config.yaml` is present.

### Results

For each experiment, the pipeline reports:

- Training/validation/test metrics: loss, plus task-appropriate scores 
- CPU and GPU training time, with a speedup ratio when a GPU is available
- A final summary table across all experiments
- Training history and prediction plots, saved under `results/`
- CSV exports of both the summary table and the full per-split metrics (`results_summary.csv`, `results_metrics.csv`)

### A note on the CPU vs GPU numbers

This project measures training time on the hardware it happens to run on. It isn't a controlled hardware benchmark, so treat the speedup numbers as illustrative rather than authoritative:

- Each experiment runs once per device; there are no repeated trials or reported variance.
- No random seed is fixed, so the CPU and GPU runs don't follow identical training trajectories (data shuffling and dropout diverge between them).
- There's no dedicated warm-up pass excluded from the clock, so GPU timing includes first-batch overhead (cuDNN autotuning, memory allocation).
- Early stopping means CPU and GPU can stop at different epoch counts, so part of any time difference may reflect that rather than raw hardware speed.

If you need rigorous numbers, add a fixed seed, a few untimed warm-up iterations, and average several runs per device.