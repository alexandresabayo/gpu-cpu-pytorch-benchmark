# PyTorch GPU vs CPU Training Comparison

A comprehensive benchmarking project comparing training performance between CPU and GPU for various neural network architectures on different datasets.

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
cd pytorch-gpu-cpu-benchmark
conda env create -n pytorch python=3.10.12
conda activate pytorch
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
cd pytorch-gpu-cpu-benchmark
conda activate pytorch
python main.py # Run the complete experiment
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

Training parameters can be configured in `config/config.yaml`.  
You can also configure parameters in the `TrainingConfig` class.

- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `train_ratio`, `val_ratio`, `test_ratio`: Data split ratios
- `dropout`: Dropout rate
- `training_history`: Whether to plot training history
- `show_progress`: Whether to show tqdm progress bars during training
- `early_stopping`: Whether to use early stopping
- `patience`: Patience for early stopping

**Note**: YAML configuration takes precedence over default values when `config/config.yaml` is present.

## Results

The script provides:

- Detailed training metrics for each experiment
- CPU vs GPU performance comparison
- Speedup calculations
- Final summary table with all results
- Visualizations of training progress and predictions

## Project Structure

```
pytorch-comparison/
│
├── config/
│   ├── config.example.yaml
│   └── config.yaml
│
├── datasets/
│   ├── temperature_inputs.csv
│   ├── temperature_labels.csv
│   ├── mnist_digit_inputs.csv
│   └── mnist_digit_labels.csv
│
├── src/                          
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   ├── conv1d.py
│   │   ├── conv2d.py
│   │   └── lstm.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loading.py
│   │   └── processing.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── experiment.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   └── visualization/    
│       ├── __init__.py
│       ├── training.py
│       └── predictions.py
│
├── main.py
├── requirements.txt
└── README.md
```