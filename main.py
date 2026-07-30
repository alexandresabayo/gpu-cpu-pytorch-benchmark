"""
Main execution script for PyTorch Testbed

This script runs the complete experiment comparing training performance
between CPU and GPU for various neural network architectures.
"""


import torch
import copy
from datetime import datetime
from rich.table import Table
from src.models import LinearBased, Conv1dBased, Conv2dBased, LSTMBased
from src.data import load_as_tensor, prepare_dataloaders, ensure_datasets_exist
from src.data.generation import cleanup_intermediate_files
from src.training import run_experiment
from src.utils import load_config_from_yaml, export_results_csv
from src.visualization import visualize_predictions
from src.visualization.training import print_metrics_summary
from src.utils.helpers import format_time
from src.richlog import Logger


def main():
    """Run the complete experiment"""
    # Timestamp shared by results/<run_timestamp>/... (plots) and
    # logs/run_<run_timestamp>.log (this run's log file) so the two trees
    # line up even though they live in separate directories.
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%M')

    with Logger(log_dir="logs", run_timestamp=run_timestamp) as log:
        results = []

        with log.step("Setup") as step:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            step.info(f"using device: {device_name}")
            if torch.cuda.is_available():
                step.info(f"GPU: {torch.cuda.get_device_name(0)}")
                step.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            step.info(f"run timestamp: {run_timestamp}")

            config, temp, mnist, dataset_config = load_config_from_yaml()

            # Downloading from raw sources if needed
            dataset_paths = ensure_datasets_exist(
                dataset_config.dataset_dir,
                auto_download=dataset_config.auto_download
            )

            cleanup_intermediate_files(dataset_config.dataset_dir)

            # Load datasets with memory efficiency
            X_temp = load_as_tensor(dataset_paths['temperature_inputs'], (-1, 84, 1))
            y_temp = load_as_tensor(dataset_paths['temperature_labels'])
            X_mnist = load_as_tensor(dataset_paths['mnist_inputs'], (-1, 1, 28, 28))
            y_mnist = load_as_tensor(dataset_paths['mnist_labels'])
            step.info("datasets loaded")

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

        # Run experiments in a loop. Each experiment is its own top-level
        # step, containing "Training on CPU" / "Training on GPU" as children
        # (opened inside run_training_experiment).
        for i, experiment in enumerate(experiments, 1):
            name = f'{i}. {experiment["name"]}'

            with log.step(name) as step:
                X_data, y_data = experiment['dataset']
                loaders = prepare_dataloaders(X_data, y_data, config)

                model_cpu = experiment['model_class'](*experiment['model_args'])
                model_gpu = copy.deepcopy(model_cpu)
                criterion = experiment['criterion']

                result = run_experiment(name, model_cpu, model_gpu, loaders, criterion, config, run_timestamp)

                visualize_predictions(name, model_gpu, loaders['test'], experiment['visualize_samples'],
                                      save_dir='results', run_timestamp=run_timestamp)

            results.append(result)

        # Final Results Summary — a report, not a log: printed once, in full,
        # after every step has closed. log.block() keeps it in the file log
        # too, without pretending it belongs to any particular step.
        table = Table(title="\nFinal Results Summary")
        table.add_column("Experiment")
        table.add_column("Params", justify="right")
        table.add_column("CPU Time", justify="right")
        table.add_column("GPU Time", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Test Metric", justify="right")

        for result in results:
            cpu_time = result['cpu_time']
            gpu_time = result['gpu_time']

            cpu_str = format_time(cpu_time)
            gpu_str = format_time(gpu_time) if gpu_time else 'N/A'
            speedup = f"x{cpu_time/gpu_time:.2f}" if gpu_time else 'N/A'

            # Get the primary metric
            test_metrics = result['gpu_metrics']['test'] if result['gpu_metrics'] else result['cpu_metrics']['test']
            metric_key = 'accuracy' if 'accuracy' in test_metrics else 'r2'
            metric = f'{test_metrics[metric_key]:.3f} ({metric_key[:3]})'

            table.add_row(
                result['name'], f"{result['n_params']:,}", cpu_str, gpu_str, speedup, metric
            )

        log.block(table)

        for result in results:
            metrics = result['gpu_metrics'] or result['cpu_metrics']
            header = f"\n{result['name']} (params: {result['n_params']:,}):"
            log.block(header, indent=3)
            print_metrics_summary(metrics)
        
        export_results_csv(results, save_dir='results', run_timestamp=run_timestamp)


if __name__ == '__main__':
    main()
