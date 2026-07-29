"""Data loading utilities"""

import pandas as pd
import torch
from typing import Optional
from collections.abc import Generator


def load_as_tensor(csv_path: str, reshape: Optional[tuple] = None) -> torch.Tensor:
    """Load CSV and convert to float32 tensor with optional reshaping"""
    data = pd.read_csv(csv_path, header=None).values.astype('float32')
    tensor_data = torch.from_numpy(data)
    return tensor_data.reshape(reshape) if reshape else tensor_data


def load_as_tensor_streaming(csv_path: str,
                             chunk_size: int = 10000) -> Generator[torch.Tensor, None, None]:
    """Streaming data loader that yields chunks of data"""
    with open(csv_path, 'r') as f:
        header = f.readline()  # Skip header

        while True:
            chunk_data = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:  # End of file
                    break
                line = line.strip()
                if line:  # Skip empty lines
                    chunk_data.append([float(x) for x in line.split(',')])

            if not chunk_data:  # No more data
                break

            yield torch.tensor(chunk_data, dtype=torch.float32)
