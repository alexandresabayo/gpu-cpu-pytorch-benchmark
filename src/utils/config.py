"""Configuration classes for experiments"""

import yaml
from dataclasses import dataclass
from ..richlog import StepHandle

# Define dataclasses first to avoid circular references
@dataclass
class TrainingConfig:
    """Centralized configuration for training experiments"""
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 0.001
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    dropout: float = 0.2
    training_history: bool = True
    early_stopping : bool = True
    patience: int = 20
    
@dataclass
class TemperatureConfig:
    """Temperature time series specification"""
    seq_length: int = 84        # Historical timesteps
    pred_length: int = 12       # Prediction horizon
    features: int = 1           # Number of features per timestep

@dataclass
class MNISTConfig:
    """MNIST dataset specification"""
    channels: int = 1
    height: int = 28
    width: int = 28
    num_classes: int = 10


@dataclass
class DatasetConfig:
    """Dataset configuration for loading and auto-download"""
    dataset_dir: str = "datasets"
    auto_download: bool = True
    temperature_inputs: str = "datasets/temperature_inputs.csv"
    temperature_labels: str = "datasets/temperature_labels.csv"
    mnist_inputs: str = "datasets/mnist_digit_inputs.csv"
    mnist_labels: str = "datasets/mnist_digit_labels.csv"


def load_config_from_yaml(step: StepHandle) -> tuple[TrainingConfig, TemperatureConfig, MNISTConfig, DatasetConfig]:
    """Load configuration from YAML file for all dataclasses.

    Args:
        step: the open StepHandle to log warnings into if config.yaml is
            missing, invalid, or fails to load for any other reason.
            Defaults are used in all three cases; only the visibility of
            the fallback has changed, not the fallback itself.
    """
    # Default configurations
    training = TrainingConfig()
    temperature = TemperatureConfig()
    mnist = MNISTConfig()
    dataset = DatasetConfig()
    
    try:
        with open('config/config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Load TrainingConfig from YAML
        training_config = yaml_config.get('training', {})
        training = TrainingConfig(
            batch_size=training_config.get('batch_size', 256),
            epochs=training_config.get('epochs', 200),
            learning_rate=training_config.get('learning_rate', 0.001),
            train_ratio=training_config.get('train_ratio', 0.7),
            val_ratio=training_config.get('val_ratio', 0.2),
            test_ratio=training_config.get('test_ratio', 0.1),
            dropout=training_config.get('dropout', 0.2),
            training_history=training_config.get('training_history', True),
            early_stopping=training_config.get('early_stopping', True),
            patience=training_config.get('patience', 20)
        )
        
        # Load TemperatureConfig from YAML
        temp_config = yaml_config.get('models', {}).get('temperature', {})
        temperature = TemperatureConfig(
            seq_length=temp_config.get('seq_length', 84),
            pred_length=temp_config.get('pred_length', 12),
            features=temp_config.get('features', 1)
        )
        
        # Load MNISTConfig from YAML
        mnist_config = yaml_config.get('models', {}).get('mnist', {})
        mnist = MNISTConfig(
            channels=mnist_config.get('channels', 1),
            height=mnist_config.get('height', 28),
            width=mnist_config.get('width', 28),
            num_classes=mnist_config.get('num_classes', 10)
        )
        
        # Load DatasetConfig from YAML
        dataset_config = yaml_config.get('datasets', {})
        dataset = DatasetConfig(
            dataset_dir=dataset_config.get('dataset_dir', 'datasets'),
            auto_download=dataset_config.get('auto_download', True),
            temperature_inputs=dataset_config.get('temperature_inputs', 'datasets/temperature_inputs.csv'),
            temperature_labels=dataset_config.get('temperature_labels', 'datasets/temperature_labels.csv'),
            mnist_inputs=dataset_config.get('mnist_inputs', 'datasets/mnist_digit_inputs.csv'),
            mnist_labels=dataset_config.get('mnist_labels', 'datasets/mnist_digit_labels.csv')
        )
        
        step.info("loaded config/config.yaml")

    except FileNotFoundError:
        step.warn("config/config.yaml not found, using default configuration")
    except yaml.YAMLError as e:
        step.warn(f"config/config.yaml is invalid YAML ({e}), using default configuration")
    except Exception as e:
        step.warn(f"failed to load config/config.yaml ({e}), using default configuration")

    return training, temperature, mnist, dataset