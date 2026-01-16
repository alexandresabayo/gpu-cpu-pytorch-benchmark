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
cd dl-pytorch-gpu-cpu-benchmark
conda env create -n pytorch python=3.10.12
conda activate pytorch
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
cd dl-pytorch-gpu-cpu-benchmark
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

## TODO: Dataset generation directly from sources

Remove local CSV dependencies - Stop loading datasets from local files
Implement dataset generation from original sources - Download and build datasets programmatically from authoritative sources (e.g., Max Planck
Institute for weather data, Yann LeCun's server for MNIST) so any developer can reproduce the exact dataset by running the code

## TODO: Implement a Proper Logging SystemS

Implement a true logging system using **Rich**, with the following characteristics:

The logger should be a **collapsible, multi-step logger** that mimics Docker-style build output. It should display hierarchical progress through complex workflows without overwhelming the terminal.

Each step is shown as a **bold title** describing the main action. While the step is running, **indented sub-operations** display detailed logs, execution timings, warnings, and progress bars in real time.

The key feature is **log collapsing**: when a step finishes, its detailed sub-logs collapse, leaving only the main step title with the **total elapsed time displayed right-aligned**. This creates a clean, readable output that shows detailed activity for the current step while preserving a concise summary of completed steps.

The logger should use consistent visual markers at the beginning of each line to convey meaning instantly:

* `[+]` SUCCESS — completed successfully
* `[-]` FAILURE — failed operation
* `[!]` WARNING — suspicious or risky condition
* `[x]` ERROR — fatal error requiring abort
* `[?]` PROMPT — user input required
* `[~]` RUNNING — operation in progress
* *(no marker)* INFO — neutral, informational messages

INFO messages are the default output and should not include any prefix marker. Their meaning should be conveyed through **indentation, placement within a step, and subtle styling** rather than explicit symbols. An INFO message may only appear under an active step; in other words, it must be part of a sub-operation.

**Visual styling:** The logger should use a **minimal color scheme**, primarily white text with **bold** and **dim** styles to create visual hierarchy. INFO messages should appear as slightly dim text. Standard status messages and success indicators should avoid bright colors. However, **errors, warnings, and failures** should retain their conventional colors (red for errors, yellow/orange for warnings) so critical issues remain immediately visible.

In addition to terminal output, the system must automatically generate log files. Each run should create a dedicated log file; every entry must be timestamped and stored in a configurable log directory. Log filenames should include the execution date and time for straightforward log management. File logs must preserve all detailed output, including the sub-steps that are collapsed in the terminal view.

**Note:** Log files should follow the established marker convention. Do not add redundant markers like `[WARNING]` when `[!]` is already present on the same line. INFO entries may be stored with an internal log level but should not introduce additional visual prefixes.

**Implementation goal:** Replace all existing `print()` statements in the code (where defined) with proper logging calls. This includes replacing or converting progress bars (e.g., `tqdm`), console prints, and other direct terminal writes to use the centralized logging system.

**Technical requirements:**

* The logger must be **robust against terminal window resizing**. The display should adapt gracefully if the terminal is resized multiple times during execution, without breaking the layout or losing information.
* Sub-operations should be **indented automatically** by the logging system based on hierarchy level. Developers should not need to manually add spaces to message strings — the logger handles indentation internally.

This logging style is ideal for **build systems, training pipelines, deployment scripts, pentesting tools, or any multi-phase workflow** where live feedback is needed during execution but a clean, minimal summary is preferred once steps complete. It combines the advantages of verbose logging (visibility into current actions) with minimal visual clutter (collapsed completed steps).

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