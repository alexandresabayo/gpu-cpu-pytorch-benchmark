"""1D CNN for time series data"""

import torch
import torch.nn as nn


class Conv1dBased(nn.Module):
    """1D CNN for time series data"""
    
    def __init__(self, in_channels: int, output_size: int, dropout: float = 0.2):
        super(Conv1dBased, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose from (batch, timesteps, features) to (batch, features, timesteps)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x)