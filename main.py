"""
Main execution script for PyTorch GPU vs CPU Training Comparison

This script runs the complete experiment comparing training performance
between CPU and GPU for various neural network architectures.
"""


import torch
import copy
from datetime import datetime
from src.models import LinearBased, Conv1dBased, Conv2dBased, LSTMBased
from src.data import load_as_tensor, prepare_dataloaders
from src.training import run_experiment
from src.utils import load_config_from_yaml
from src.visualization import visualize_predictions
from src.utils.helpers import format_time


def main():
    """Run the complete experiment"""
    print(f'Using device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Create timestamp for this run (used for organizing all plots from this execution)
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%M')
    print(f'Run timestamp: {run_timestamp}')
    
    results = []
    config, temp, mnist = load_config_from_yaml()

    # Load datasets with memory efficiency
    print("Loading datasets... ", end='')
    X_temp = load_as_tensor('datasets/temperature_inputs.csv', (-1, 84, 1))
    y_temp = load_as_tensor('datasets/temperature_labels.csv')
    X_mnist = load_as_tensor('datasets/mnist_digit_inputs.csv', (-1, 1, 28, 28))
    y_mnist = load_as_tensor('datasets/mnist_digit_labels.csv')
    print("Done!")

    # Define experiments configuration
    experiments = [
        # Temperature experiments
        {
            'name': 'Temperature - MLP',
            'dataset': (X_temp, y_temp),
            'model_class': LinearBased,
            'model_args': (temp.seq_length * temp.features, [516, 256, 128], temp.pred_length),
            'criterion': torch.nn.MSELoss(),
            'visualize_samples': 3
        },
        {
            'name': 'Temperature - CNN',
            'dataset': (X_temp, y_temp),
            'model_class': Conv1dBased,
            'model_args': (temp.features, temp.pred_length),
            'criterion': torch.nn.MSELoss(),
            'visualize_samples': 3
        },
        {
            'name': 'Temperature - LSTM',
            'dataset': (X_temp, y_temp),
            'model_class': LSTMBased,
            'model_args': (temp.features, 128, 2, temp.pred_length),
            'criterion': torch.nn.MSELoss(),
            'visualize_samples': 3
        },
        # MNIST experiments
        {
            'name': 'MNIST - MLP',
            'dataset': (X_mnist, y_mnist),
            'model_class': LinearBased,
            'model_args': (mnist.height * mnist.width, [256, 128, 64], mnist.num_classes, 0.4),
            'criterion': torch.nn.CrossEntropyLoss(),
            'visualize_samples': 5
        },
        {
            'name': 'MNIST - CNN',
            'dataset': (X_mnist, y_mnist),
            'model_class': Conv2dBased,
            'model_args': (mnist.channels, mnist.num_classes),
            'criterion': torch.nn.CrossEntropyLoss(),
            'visualize_samples': 5
        },
        {
            'name': 'MNIST - LSTM',
            'dataset': (X_mnist, y_mnist),
            'model_class': LSTMBased,
            'model_args': (mnist.width, 128, 2, mnist.num_classes),
            'criterion': torch.nn.CrossEntropyLoss(),
            'visualize_samples': 5
        }
    ]

    # Run experiments in a loop
    for i, experiment in enumerate(experiments, 1):
        X_data, y_data = experiment['dataset']
        loaders = prepare_dataloaders(X_data, y_data, config)
        
        model_cpu = experiment['model_class'](*experiment['model_args'])
        model_gpu = copy.deepcopy(model_cpu)
        criterion = experiment['criterion']
        
        name = f'{i}. {experiment["name"]}'
        result = run_experiment(name, model_cpu, model_gpu, loaders, criterion, config, run_timestamp)
        
        visualize_predictions(name, model_gpu, loaders['test'], experiment['visualize_samples'], 
                              save_dir='results', run_timestamp=run_timestamp)
        
        results.append(result)

    # Final Results Summary
    print(f'\n{"="*80}')
    print(f'FINAL RESULTS SUMMARY')
    print(f'{"="*80}')
    print(
        f'{"Experiment":<21} {"Params":<10} {"CPU Time":<11} '
        f'{"GPU Time":<11} {"Speedup":<10} {"Test Metric":<12}'
    )
    print(f'{"-"*80}')

    for result in results:
        exp_name = result['name']
        n_params = result['n_params']
        cpu_time = result['cpu_time']
        gpu_time = result['gpu_time']
        
        cpu_str = format_time(cpu_time)
        gpu_str = format_time(gpu_time) if gpu_time else 'N/A'
        speedup = f"x{cpu_time/gpu_time:.2f}" if gpu_time else 'N/A'

        # Get the primary metric  
        test_metrics = result['gpu_metrics']['test'] or result['cpu_metrics']['test']
        metric_key = 'accuracy' if 'accuracy' in test_metrics else 'r2'
        metric = f'{test_metrics[metric_key]:.3f} ({metric_key[:3]})'

        print(
            f'{exp_name:<21} {n_params:<10,} {cpu_str:<11} '
            f'{gpu_str:<11} {speedup:<10} {metric:<12}'
        )

    print(f'{"-"*80}')

    # Lazy import to avoid circular dependencies
    from src.visualization.training import print_metrics_summary
    
    print(f'\n{"="*80}')
    print('MODEL PERFORMANCE DETAILS')
    print(f'{"="*80}')

    for result in results:
        metrics = result['gpu_metrics'] or result['cpu_metrics']

        print(f'\n{result["name"]} (params: {result["n_params"]:,}):\n')
        print_metrics_summary(metrics)
        print(f'\n{"-"*80}')


if __name__ == '__main__':
    main()