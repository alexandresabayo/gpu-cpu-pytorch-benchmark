"""Configuration classes for experiments"""

import yaml
from dataclasses import dataclass, fields
from ..richlog import current

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


def _from_dict(cls, data: dict):
    """Build a dataclass instance from a dict, ignoring unknown keys"""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config_from_yaml() -> tuple[TrainingConfig, TemperatureConfig, MNISTConfig, DatasetConfig]:
    """Load configuration from YAML file for all dataclasses."""
    step = current()

    # Default configurations
    training = TrainingConfig()
    temperature = TemperatureConfig()
    mnist = MNISTConfig()
    dataset = DatasetConfig()

    try:
        with open('config/config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)

        training = _from_dict(TrainingConfig, yaml_config.get('training', {}))
        temperature = _from_dict(TemperatureConfig, yaml_config.get('models', {}).get('temperature', {}))
        mnist = _from_dict(MNISTConfig, yaml_config.get('models', {}).get('mnist', {}))
        dataset = _from_dict(DatasetConfig, yaml_config.get('datasets', {}))

        step.info("loaded config/config.yaml")

    except FileNotFoundError:
        step.warn("config/config.yaml not found, using default configuration")
    except yaml.YAMLError as e:
        step.warn(f"config/config.yaml is invalid YAML ({e}), using default configuration")
    except Exception as e:
        step.warn(f"failed to load config/config.yaml ({e}), using default configuration")

    return training, temperature, mnist, dataset
