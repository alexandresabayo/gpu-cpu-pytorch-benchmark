"""LSTM for time series prediction/classification"""

import torch
import torch.nn as nn


class LSTMBased(nn.Module):
    """LSTM for time series prediction/classification"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMBased, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch, channels, height, width = x.shape
            x = x.view(batch, height, channels * width)  # Treat rows as sequence
        
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])