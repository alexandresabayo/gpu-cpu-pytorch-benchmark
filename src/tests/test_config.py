"""Tests for configuration module"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils.config import TrainingConfig, TemperatureConfig, MNISTConfig, load_config_from_yaml


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.batch_size == 256
        assert config.epochs == 200
        assert config.learning_rate == 0.001
        assert config.num_workers == 4
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.2
        assert config.test_ratio == 0.1
        assert config.dropout == 0.2
        assert config.training_history is True
        assert config.show_progress is False
        assert config.early_stopping is True
        assert config.patience == 20

    def test_custom_values(self):
        """Test configuration with custom values"""
        config = TrainingConfig(
            batch_size=128,
            epochs=100,
            learning_rate=0.01,
            dropout=0.5,
            patience=10
        )
        
        assert config.batch_size == 128
        assert config.epochs == 100
        assert config.learning_rate == 0.01
        assert config.dropout == 0.5
        assert config.patience == 10

    def test_data_split_ratios_sum(self):
        """Test that data split ratios sum to 1.0"""
        config = TrainingConfig()
        total = config.train_ratio + config.val_ratio + config.test_ratio
        assert abs(total - 1.0) < 1e-6

    def test_positive_values(self):
        """Test that numeric values are positive"""
        config = TrainingConfig()
        
        assert config.batch_size > 0
        assert config.epochs > 0
        assert config.learning_rate > 0
        assert config.num_workers >= 0
        assert config.dropout >= 0
        assert config.patience >= 0


class TestTemperatureConfig:
    """Test TemperatureConfig dataclass"""

    def test_default_values(self):
        """Test default temperature configuration"""
        config = TemperatureConfig()
        
        assert config.seq_length == 84
        assert config.pred_length == 12
        assert config.features == 1

    def test_custom_values(self):
        """Test temperature configuration with custom values"""
        config = TemperatureConfig(seq_length=100, pred_length=20, features=2)
        
        assert config.seq_length == 100
        assert config.pred_length == 20
        assert config.features == 2

    def test_positive_values(self):
        """Test that all values are positive"""
        config = TemperatureConfig()
        
        assert config.seq_length > 0
        assert config.pred_length > 0
        assert config.features > 0


class TestMNISTConfig:
    """Test MNISTConfig dataclass"""

    def test_default_values(self):
        """Test default MNIST configuration"""
        config = MNISTConfig()
        
        assert config.channels == 1
        assert config.height == 28
        assert config.width == 28
        assert config.num_classes == 10

    def test_custom_values(self):
        """Test MNIST configuration with custom values"""
        config = MNISTConfig(channels=3, height=32, width=32, num_classes=100)
        
        assert config.channels == 3
        assert config.height == 32
        assert config.width == 32
        assert config.num_classes == 100

    def test_positive_values(self):
        """Test that all values are positive"""
        config = MNISTConfig()
        
        assert config.channels > 0
        assert config.height > 0
        assert config.width > 0
        assert config.num_classes > 0

    def test_standard_mnist_dimensions(self):
        """Test standard MNIST dimensions"""
        config = MNISTConfig()
        image_pixels = config.height * config.width
        assert image_pixels == 784  # Standard MNIST


class TestLoadConfigFromYaml:
    """Test YAML configuration loading"""

    def test_load_default_when_file_not_found(self):
        """Test that function handles missing config.yaml gracefully"""
        # Change to a directory without config.yaml
        import os
        import sys
        original_cwd = os.getcwd()
        
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Should return default configs, not raise an error (bug fixed)
                training, temp, mnist = load_config_from_yaml()
                
                # Verify we got valid config objects
                assert isinstance(training, TrainingConfig)
                assert isinstance(temp, TemperatureConfig)
                assert isinstance(mnist, MNISTConfig)
                
                # Verify default values
                assert training.epochs == 200
                assert temp.seq_length == 84
                assert mnist.num_classes == 10
        finally:
            os.chdir(original_cwd)

    def test_load_custom_training_config(self):
        """Test loading custom training configuration from YAML"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Create config.yaml
                config_data = {
                    'training': {
                        'batch_size': 128,
                        'epochs': 100,
                        'learning_rate': 0.01,
                        'dropout': 0.3,
                        'patience': 15
                    },
                    'models': {
                        'temperature': {},
                        'mnist': {}
                    }
                }
                
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(config_data, f)
                
                training, temp, mnist = load_config_from_yaml()
                
                assert training.batch_size == 128
                assert training.epochs == 100
                assert training.learning_rate == 0.01
                assert training.dropout == 0.3
                assert training.patience == 15
        finally:
            os.chdir(original_cwd)

    def test_load_custom_temperature_config(self):
        """Test loading custom temperature configuration from YAML"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                config_data = {
                    'training': {},
                    'models': {
                        'temperature': {
                            'seq_length': 120,
                            'pred_length': 24,
                            'features': 2
                        },
                        'mnist': {}
                    }
                }
                
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(config_data, f)
                
                training, temp, mnist = load_config_from_yaml()
                
                assert temp.seq_length == 120
                assert temp.pred_length == 24
                assert temp.features == 2
        finally:
            os.chdir(original_cwd)

    def test_load_custom_mnist_config(self):
        """Test loading custom MNIST configuration from YAML"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                config_data = {
                    'training': {},
                    'models': {
                        'temperature': {},
                        'mnist': {
                            'channels': 3,
                            'height': 32,
                            'width': 32,
                            'num_classes': 100
                        }
                    }
                }
                
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(config_data, f)
                
                training, temp, mnist = load_config_from_yaml()
                
                assert mnist.channels == 3
                assert mnist.height == 32
                assert mnist.width == 32
                assert mnist.num_classes == 100
        finally:
            os.chdir(original_cwd)

    def test_partial_config_uses_defaults(self):
        """Test that missing config keys use default values"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Minimal config
                config_data = {
                    'training': {
                        'batch_size': 512
                    },
                    'models': {
                        'temperature': {},
                        'mnist': {}
                    }
                }
                
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(config_data, f)
                
                training, temp, mnist = load_config_from_yaml()
                
                # Custom value
                assert training.batch_size == 512
                # Default values for not specified fields
                assert training.epochs == 200
                assert training.learning_rate == 0.001
        finally:
            os.chdir(original_cwd)

    def test_invalid_yaml_uses_defaults(self):
        """Test that defaults are used when YAML is invalid (bug fixed)"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Write invalid YAML
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    f.write("{ invalid yaml: [")
                
                # Should return defaults, not raise error (bug fixed)
                training, temp, mnist = load_config_from_yaml()
                
                # Verify we got default configs
                assert isinstance(training, TrainingConfig)
                assert isinstance(temp, TemperatureConfig)
                assert isinstance(mnist, MNISTConfig)
                assert training.epochs == 200
        finally:
            os.chdir(original_cwd)

    def test_returns_three_configs(self):
        """Test that function returns all three config objects when valid YAML exists"""
        import os
        original_cwd = os.getcwd()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Create valid config.yaml
                config_data = {
                    'training': {},
                    'models': {
                        'temperature': {},
                        'mnist': {}
                    }
                }
                os.makedirs('config', exist_ok=True)
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(config_data, f)
                
                result = load_config_from_yaml()
                
                assert len(result) == 3
                training, temp, mnist = result
                assert isinstance(training, TrainingConfig)
                assert isinstance(temp, TemperatureConfig)
                assert isinstance(mnist, MNISTConfig)
        finally:
            os.chdir(original_cwd)
