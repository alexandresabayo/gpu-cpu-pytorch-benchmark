"""Data loading and processing utilities"""
from .loading import (
    load_as_tensor,
    load_as_tensor_streaming
)
from .processing import (
    normalize_inputs,
    prepare_labels,
    split_data,
    prepare_dataloaders
)
from .generation import (
    download_mnist_csv,
    download_jena_climate_csv,
    generate_all_datasets,
    get_dataset_paths,
    ensure_datasets_exist
)