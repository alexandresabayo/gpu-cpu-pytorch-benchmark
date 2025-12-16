"""Multi-Layer Perceptron for regression/classification"""

import torch
import torch.nn as nn
from typing import List


class LinearBased(nn.Module):
    """Multi-Layer Perceptron for regression/classification"""
    
    def __init__(self, in_features: int, hidden_sizes: List[int], output_size: int, dropout: float = 0.2):
        super(LinearBased, self).__init__()
        layers = []
        layers.append(nn.Flatten())
        
        prev_size = in_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)