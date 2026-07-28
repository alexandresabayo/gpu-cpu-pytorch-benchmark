"""Data loading utilities with memory efficiency options"""

import pandas as pd
import torch
import numpy as np
from typing import Optional
import os
from collections.abc import Generator



def load_as_tensor(csv_path: str, reshape: Optional[tuple] = None,
                   memory_efficient: bool = True) -> torch.Tensor:
    """Load CSV and convert to float32 tensor with optional reshaping"""
    if memory_efficient:
        return _load_memory_efficient(csv_path, reshape)
    else:
        # Original approach for backward compatibility
        data = pd.read_csv(csv_path, header=None).values.astype('float32')
        return torch.from_numpy(data).reshape(reshape) if reshape else torch.from_numpy(data)


def _load_memory_efficient(csv_path: str,
                           reshape: Optional[tuple] = None) -> torch.Tensor:
    """Memory-efficient loading using numpy memmap and chunked processing"""
    
    # First, determine the shape by reading just the header
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        n_columns = len(header.split(','))
    
    # Count total rows efficiently
    n_rows = sum(1 for _ in open(csv_path, 'r')) - 1  # Subtract header
    
    # Create a temporary binary file for memory mapping
    temp_file = csv_path + '.memmap'
    
    # Create memory-mapped array
    data_array = np.memmap(temp_file, dtype='float32', mode='w+',
                          shape=(n_rows, n_columns))
    
    # Process file in chunks to populate the memmap
    chunk_size = 10000  # Process 10K rows at a time
    
    with open(csv_path, 'r') as f:
        header = f.readline()  # Skip header
        
        for i in range(0, n_rows, chunk_size):
            chunk_end = min(i + chunk_size, n_rows)
            chunk_rows = chunk_end - i
            
            # Read chunk
            chunk_data = []
            for _ in range(chunk_rows):
                line = f.readline().strip()
                if line:  # Skip empty lines
                    chunk_data.append([float(x) for x in line.split(',')])
            
            # Store in memmap
            data_array[i:chunk_end] = np.array(chunk_data, dtype='float32')
    
    # Convert to tensor
    tensor_data = torch.from_numpy(data_array)
    
    # Clean up temporary file
    try:
        os.unlink(temp_file)
    except:
        pass  # File might not exist or already deleted
    
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