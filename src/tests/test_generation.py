#!/usr/bin/env python3
"""Test script for dataset generation from raw sources."""

import sys
from pathlib import Path

def test_generation():
    """Test that dataset generation produces the correct shapes."""
    print("="*80)
    print("Testing Dataset Generation")
    print("="*80)
    
    # Import after path setup
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.generation import download_mnist_csv, download_jena_climate_csv
    
    test_dir = Path("test_datasets")
    test_dir.mkdir(exist_ok=True)
    
    print("\n[1] Testing MNIST generation...")
    try:
        mnist_inputs, mnist_labels = download_mnist_csv(
            output_dir=str(test_dir),
            num_samples=42000,
            force_download=True
        )
        
        # Verify shapes
        import pandas as pd
        df_inputs = pd.read_csv(mnist_inputs, header=None)
        df_labels = pd.read_csv(mnist_labels, header=None)
        
        print(f"  Inputs shape: {df_inputs.shape}")
        print(f"  Labels shape: {df_labels.shape}")
        
        assert df_inputs.shape == (42000, 784), f"Expected (42000, 784), got {df_inputs.shape}"
        assert df_labels.shape == (42000, 1), f"Expected (42000, 1), got {df_labels.shape}"
        assert 'label' in df_labels.columns, "Labels should have 'label' column"
        
        print("  ✓ MNIST shapes match!")
        
    except Exception as e:
        print(f"  ✗ MNIST generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[2] Testing Temperature (Jena Climate) generation...")
    try:
        temp_inputs, temp_labels = download_jena_climate_csv(
            output_dir=str(test_dir),
            force_download=True,
            target_samples=34968
        )
        
        # Verify shapes
        df_inputs = pd.read_csv(temp_inputs, header=None)
        df_labels = pd.read_csv(temp_labels, header=None)
        
        print(f"  Inputs shape: {df_inputs.shape}")
        print(f"  Labels shape: {df_labels.shape}")
        
        assert df_inputs.shape == (34968, 84), f"Expected (34968, 84), got {df_inputs.shape}"
        assert df_labels.shape == (34968, 12), f"Expected (34968, 12), got {df_labels.shape}"
        
        # Verify column names
        expected_input_cols = [f"{2*(i+1)}h" for i in range(84)]
        expected_label_cols = [f"{170+2*i}h" for i in range(12)]
        
        assert list(df_inputs.columns) == expected_input_cols, "Input column names don't match"
        assert list(df_labels.columns) == expected_label_cols, "Label column names don't match"
        
        print("  ✓ Temperature shapes match!")
        
    except Exception as e:
        print(f"  ✗ Temperature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
    
    # Cleanup
    import shutil
    print("\nCleaning up test files...")
    shutil.rmtree(test_dir)
    
    return True


if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)
