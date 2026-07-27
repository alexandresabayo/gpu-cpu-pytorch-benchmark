## PyTorch GPU vs CPU training comparison

A comprehensive benchmarking project comparing training performance between CPU and GPU for various neural network architectures on different datasets.

### Features

- **Multiple model architectures**: MLP, 1D CNN, 2D CNN, LSTM
- **Multiple datasets**: Temperature time series, MNIST digits
- **Comprehensive benchmarking**: CPU vs GPU performance comparison
- **Detailed metrics**: Training time, memory usage, accuracy, loss, etc.
- **Visualization**: Training history plots, prediction visualizations
- **Modular design**: Clean separation of concerns

### Requirements

* **conda** ([download Miniforge3](https://github.com/conda-forge/miniforge)) to create and manage the Python environment.

### Quickstart

```sh
make install      # Create the Conda environment and install dependencies
make data         # Generate/download the datasets
make benchmark    # Run experiments
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

The script provides:

- Detailed training metrics for each experiment
- CPU vs GPU performance comparison
- Speedup calculations
- Final summary table with all results
- Visualizations of training progress and predictions