# PyTorch GPU vs CPU Training Comparison

A comprehensive benchmarking project comparing training performance between CPU and GPU for various neural network architectures on different datasets.

## Project Structure

```
pytorch-comparison/
├── datasets/                  # Dataset files
│   ├── temperature_inputs.csv
│   ├── temperature_labels.csv
│   ├── mnist_digit_inputs.csv
│   └── mnist_digit_labels.csv
├── src/                      # Main source code
│   ├── __init__.py
│   ├── models/           # Neural network models
│   │   ├── __init__.py
│   │   ├── linear.py     # MLP model
│   │   ├── conv1d.py     # 1D CNN for time series
│   │   ├── conv2d.py     # 2D CNN for images
│   │   └── lstm.py       # LSTM model
│   ├── data/             # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loading.py
│   │   └── processing.py
│   ├── training/         # Training logic
│   │   ├── __init__.py
│   │   ├── core.py       # Core training functions
│   │   └── experiment.py # Experiment running
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py     # Configuration classes
│   │   ├── metrics.py    # Metrics calculation
│   │   └── helpers.py    # Helper functions
│   └── visualization/    # Visualization utilities
│       ├── __init__.py
│       ├── training.py   # Training visualization
│       └── predictions.py # Prediction visualization
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── config.yaml              # Configuration file
└── README.md                # This file
```

## Features

- **Multiple Model Architectures**: MLP, 1D CNN, 2D CNN, LSTM
- **Multiple Datasets**: Temperature time series, MNIST digits
- **Comprehensive Benchmarking**: CPU vs GPU performance comparison
- **Detailed Metrics**: Training time, memory usage, accuracy, loss, etc.
- **Visualization**: Training history plots, prediction visualizations
- **Modular Design**: Clean separation of concerns

## Requirements

### Manual Setup with Conda

```bash
# Create the conda environment
conda env create -n pytorch python=3.10.12

# Activate the environment
conda activate pytorch

# Install package in development mode
pip install -r requirements.txt
```

### Manual Setup with pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Using Conda Environment

```bash
# Activate your conda environment
conda activate pytorch

# Run the complete experiment
python main.py
```

### Run specific experiments

You can modify the `main.py` script to run only specific experiments by commenting out the ones you don't need.

## Experiments

The project runs 6 experiments comparing different model architectures on different datasets:

1. **Temperature - MLP**: Multi-Layer Perceptron on temperature time series data
2. **Temperature - CNN**: 1D CNN on temperature time series data  
3. **Temperature - LSTM**: LSTM on temperature time series data
4. **MNIST - MLP**: Multi-Layer Perceptron on MNIST digit classification
5. **MNIST - CNN**: 2D CNN on MNIST digit classification
6. **MNIST - LSTM**: LSTM on MNIST digit classification

## Configuration

### YAML Configuration

Training parameters can be configured in `config.yaml`:

```yaml
training:
  batch_size: 256
  epochs: 200
  learning_rate: 0.001
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  dropout: 0.2
  training_history: true
  show_progress: true  # Set to true to see tqdm progress bars
  early_stopping: true
  patience: 20
```

### Programmatic Configuration

You can also configure parameters in the `TrainingConfig` class in `src/pytorch_comparison/utils/config.py`:

- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `train_ratio`, `val_ratio`, `test_ratio`: Data split ratios
- `dropout`: Dropout rate
- `training_history`: Whether to plot training history
- `show_progress`: Whether to show tqdm progress bars during training
- `early_stopping`: Whether to use early stopping
- `patience`: Patience for early stopping

**Note**: YAML configuration takes precedence over default values when `config.yaml` is present.

## Results

The script provides:

- Detailed training metrics for each experiment
- CPU vs GPU performance comparison
- Speedup calculations
- Final summary table with all results
- Visualizations of training progress and predictions

## License

This project is open source and available under the MIT License.