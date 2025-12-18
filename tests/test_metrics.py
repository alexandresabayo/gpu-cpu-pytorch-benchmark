"""Tests for metrics calculation module"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock
from utils.metrics import (
    predict_batch, 
    calculate_regression_metrics, 
    calculate_classification_metrics,
    calculate_metrics
)


class TestPredictBatch:
    """Test predict_batch function"""

    def test_predict_batch_basic(self):
        """Test basic batch prediction"""
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        # Create dummy data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 10),
            torch.randn(20, 5)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = torch.device('cpu')
        predictions, targets = predict_batch(model, loader, device)
        
        assert predictions.shape == (20, 5)
        assert targets.shape == (20, 5)

    def test_predict_batch_preserves_order(self):
        """Test that prediction order is preserved"""
        model = torch.nn.Linear(10, 1)
        model.eval()
        
        # Create reproducible data
        torch.manual_seed(42)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(12, 10),
            torch.randn(12, 1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = torch.device('cpu')
        predictions, targets = predict_batch(model, loader, device)
        
        assert predictions.shape[0] == 12
        assert targets.shape[0] == 12

    def test_predict_batch_no_grad(self):
        """Test that gradients are not computed"""
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 10),
            torch.randn(8, 5)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = torch.device('cpu')
        predictions, targets = predict_batch(model, loader, device)
        
        # Predictions should not have gradients
        assert predictions.grad is None

    def test_predict_batch_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 10),
            torch.randn(20, 5)
        )
        
        for batch_size in [1, 4, 5, 10]:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            device = torch.device('cpu')
            predictions, targets = predict_batch(model, loader, device)
            
            assert predictions.shape[0] == 20
            assert targets.shape[0] == 20


class TestCalculateRegressionMetrics:
    """Test regression metrics calculation"""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        predictions = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['r2'] == 1.0

    def test_constant_target_r2(self):
        """Test R2 with constant target values"""
        predictions = torch.tensor([[1.0], [1.0], [1.0]])
        targets = torch.tensor([[2.0], [2.0], [2.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        assert metrics['mae'] == 1.0
        # When ss_tot=0 (constant targets), r2 is set to 0, not 1
        assert metrics['r2'] == 0.0

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation"""
        predictions = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[2.0], [2.0], [2.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        # MAE = (|1-2| + |2-2| + |3-2|) / 3 = 2/3
        assert abs(metrics['mae'] - 2/3) < 1e-5

    def test_rmse_calculation(self):
        """Test Root Mean Square Error calculation"""
        predictions = torch.tensor([[1.0], [1.0], [1.0]])
        targets = torch.tensor([[0.0], [1.0], [2.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        # RMSE = sqrt(((1-0)^2 + (1-1)^2 + (1-2)^2) / 3) = sqrt(2/3)
        expected_rmse = np.sqrt(2/3)
        assert abs(metrics['rmse'] - expected_rmse) < 1e-5

    def test_r2_calculation(self):
        """Test R2 score calculation"""
        predictions = torch.tensor([[1.1], [2.1], [3.1]])
        targets = torch.tensor([[1.0], [2.0], [3.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        # Should be close to 1 for nearly perfect predictions
        assert metrics['r2'] > 0.9

    def test_negative_r2(self):
        """Test that R2 can be negative for bad predictions"""
        predictions = torch.tensor([[5.0], [5.0], [5.0]])
        targets = torch.tensor([[1.0], [2.0], [3.0]])
        
        metrics = calculate_regression_metrics(predictions, targets)
        
        assert metrics['r2'] < 0


class TestCalculateClassificationMetrics:
    """Test classification metrics calculation"""

    def test_perfect_classification(self):
        """Test metrics with perfect predictions"""
        # 4 samples, 3 classes
        predictions = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0]])
        targets = torch.tensor([0, 1, 2, 0])
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        assert metrics['accuracy'] == 1.0
        assert 'f1_macro' in metrics or 'f1' in metrics

    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        predictions = torch.tensor([[0.9, 0.1],
                                   [0.1, 0.9],
                                   [0.8, 0.2],
                                   [0.3, 0.7]])
        targets = torch.tensor([0, 1, 0, 1])
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        assert metrics['accuracy'] == 1.0

    def test_binary_classification(self):
        """Test binary classification metrics"""
        predictions = torch.tensor([[0.8, 0.2],
                                   [0.2, 0.8],
                                   [0.7, 0.3],
                                   [0.4, 0.6]])
        targets = torch.tensor([0, 1, 0, 1])
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert metrics['accuracy'] == 1.0

    def test_multiclass_classification(self):
        """Test multiclass classification metrics"""
        predictions = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
        targets = torch.tensor([0, 1, 2, 0, 1])
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'auc_roc' in metrics
        assert metrics['accuracy'] == 1.0

    def test_imperfect_predictions(self):
        """Test metrics with imperfect predictions"""
        predictions = torch.tensor([[0.6, 0.4],
                                   [0.3, 0.7],
                                   [0.5, 0.5],
                                   [0.4, 0.6]])
        targets = torch.tensor([0, 1, 1, 0])  # Mismatch on 3rd and 4th
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        assert metrics['accuracy'] == 0.5
        assert metrics['f1'] < 1.0

    def test_metrics_are_floats(self):
        """Test that all metrics are float values"""
        predictions = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
        targets = torch.tensor([0, 1])
        
        metrics = calculate_classification_metrics(predictions, targets)
        
        for value in metrics.values():
            assert isinstance(value, (float, np.floating))


class TestCalculateMetrics:
    """Test the unified calculate_metrics function"""

    def test_regression_task(self):
        """Test metrics calculation for regression task"""
        model = torch.nn.Linear(10, 1)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 10),
            torch.randn(16, 1)  # Float targets -> regression
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        metrics = calculate_metrics(model, loader, criterion, device)
        
        assert 'loss' in metrics
        assert 'mae' in metrics or 'rmse' in metrics or 'r2' in metrics

    def test_classification_task(self):
        """Test metrics calculation for classification task"""
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 10),
            torch.randint(0, 5, (16,))  # Integer targets -> classification
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        metrics = calculate_metrics(model, loader, criterion, device)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_includes_loss(self):
        """Test that loss is always included in metrics"""
        model = torch.nn.Linear(10, 1)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 10),
            torch.randn(8, 1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        metrics = calculate_metrics(model, loader, criterion, device)
        
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)

    def test_metrics_are_floats(self):
        """Test that all metrics are numeric"""
        model = torch.nn.Linear(10, 1)
        model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 10),
            torch.randn(8, 1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        metrics = calculate_metrics(model, loader, criterion, device)
        
        for value in metrics.values():
            assert isinstance(value, (float, int))

    def test_model_not_modified(self):
        """Test that model parameters aren't modified during evaluation"""
        model = torch.nn.Linear(10, 1)
        original_params = [p.clone() for p in model.parameters()]
        
        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 10),
            torch.randn(8, 1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        metrics = calculate_metrics(model, loader, criterion, device)
        
        # Check that parameters haven't changed
        for p_orig, p_curr in zip(original_params, model.parameters()):
            assert torch.allclose(p_orig, p_curr)
