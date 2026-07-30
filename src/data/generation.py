"""Dataset generation from raw sources.

This module downloads and generates datasets directly from authoritative
sources, so no pre-processed local CSV files are required.

Supported datasets:
- MNIST: downloaded from Yann LeCun's server via torchvision. Both the
  train and test splits are combined, since nothing about this module's
  usage depends on the original train/test boundary; that split is left
  for whoever consumes the CSV to make. Sample count defaults to the full
  70,000 available, not just the 60,000-image train split.
- Temperature: Jena Climate 2009-2016 dataset from TensorFlow. The raw
  data is recorded every 10 minutes, but the windows this module produces
  are labeled in 2-hour steps (2h, 4h, ..., 168h for inputs; 170h, ...,
  192h for labels), so the series is decimated to 2-hour resolution
  before windowing; see `download_jena_climate_csv` for why building
  windows straight from the 10-minute rows would be wrong. Because of that
  resolution change, the number of available windows (~34,951) is much
  smaller than the raw row count would suggest.

Neither generator imposes a sample cap below what the source data
supports by default; both accept an optional argument to request fewer
samples, but leaving it unset returns everything available.

Every function below logs into whichever Step is currently open via
richlog.current(); none of these functions open their own top-level step,
except generate_all_datasets/get_dataset_paths/ensure_datasets_exist which
open nested child steps for the two datasets they manage.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional
import io
import time
import shutil
import contextlib

import numpy as np
import pandas as pd
import torch

from ..richlog import current

DATASETS = {
    "mnist": {"inputs": "mnist_digit_inputs.csv", "labels": "mnist_digit_labels.csv"},
    "temperature": {"inputs": "temperature_inputs.csv", "labels": "temperature_labels.csv"},
}


def download_mnist_csv(
    output_dir: str = "datasets",
    num_samples: Optional[int] = None,
    force_download: bool = False,
) -> Tuple[str, str]:
    """Download MNIST dataset and save as CSV files.

    Downloads the official MNIST dataset using torchvision (both the train
    and test splits), extracts `num_samples` samples, and saves them as CSV
    files matching the expected format.

    Args:
        output_dir: Directory to save CSV files (default: "datasets")
        num_samples: Number of samples to extract. If None (default), ALL
            available samples are used; 70,000 total (60,000 train +
            10,000 test). Pass an int to cap it lower if you want.
        force_download: If True, re-download even if files exist

    Returns:
        Tuple of (inputs_csv_path, labels_csv_path)

    Raises:
        ImportError: If torchvision is not installed
    """
    step = current()

    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
    except ImportError:
        step.error("torchvision is not installed; required for MNIST download")
        raise ImportError(
            "torchvision is required for MNIST download. "
            "Install it with: pip install torchvision"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = output_dir / DATASETS["mnist"]["inputs"]
    labels_path = output_dir / DATASETS["mnist"]["labels"]

    # Check if files already exist
    if inputs_path.exists() and labels_path.exists() and not force_download:
        step.info(f"MNIST CSV files already exist in {output_dir}")
        return str(inputs_path), str(labels_path)

    start_time = time.time()

    # torchvision uses tqdm internally for the download bar, which writes
    # to stderr; suppressed here rather than surfaced, since byte-level
    # download progress isn't part of this logger's scope.
    with contextlib.redirect_stderr(io.StringIO()):
        train_data = MNIST(
            root=str(output_dir / ".mnist_cache"),
            train=True,
            download=True,
            transform=ToTensor()
        )
        step.info("train split ready")
        test_data = MNIST(
            root=str(output_dir / ".mnist_cache"),
            train=False,
            download=True,
            transform=ToTensor()
        )
        step.info("test split ready")

    all_images = torch.cat([train_data.data, test_data.data], dim=0).numpy()
    all_labels = torch.cat([train_data.targets, test_data.targets], dim=0).numpy()

    total_available = all_images.shape[0]  # 70,000
    step.info(f"{total_available} total samples available")

    if num_samples is None:
        num_samples = total_available
    else:
        num_samples = min(num_samples, total_available)

    images = all_images[:num_samples]
    labels_arr = all_labels[:num_samples]

    # Flatten images: (N, 28, 28) -> (N, 784)
    images_flat = images.reshape(num_samples, -1).astype('float32')

    # Save inputs CSV
    inputs_df = pd.DataFrame(images_flat)
    inputs_df.to_csv(inputs_path, index=False, header=False)
    step.info(f"saved inputs: {inputs_path} ({images_flat.shape[0]} samples, {images_flat.shape[1]} features)")

    # Save labels CSV with header
    labels_df = pd.DataFrame(labels_arr, columns=['label'])
    labels_df.to_csv(labels_path, index=False, header=False)
    step.info(f"saved labels: {labels_path} ({labels_arr.shape[0]} samples)")

    elapsed = time.time() - start_time
    step.info(f"completed in {elapsed:.2f}s")

    return str(inputs_path), str(labels_path)


def download_jena_climate_csv(
    output_dir: str = "datasets",
    force_download: bool = False,
    target_samples: Optional[int] = None,
) -> Tuple[str, str]:
    """Download Jena Climate dataset and save as CSV files.

    Downloads the Jena Climate 2009-2016 dataset, processes it into sliding
    windows for time series forecasting, and saves as CSV files matching
    the expected format.

    The Jena Climate dataset contains 14 weather features measured every
    10 minutes from 2009-2016. This function extracts only the temperature
    column, performs linear interpolation to fill any missing values,
    and creates windows to match the expected CSV shape.

    Args:
        output_dir: Directory to save CSV files (default: "datasets")
        force_download: If True, re-download even if files exist
        target_samples: Number of sliding-window samples to generate. If
            None (default), ALL available windows at 2-hour resolution are
            used (len(series) - window_size + 1, ~34,951 samples for the
            full 2009-2016 series). Pass an int to cap it lower if you want.

    Returns:
        Tuple of (inputs_csv_path, labels_csv_path)

    Raises:
        ValueError: If temperature column not found, or if an explicit
            target_samples is requested that exceeds what's available.
    """
    step = current()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = output_dir / DATASETS["temperature"]["inputs"]
    labels_path = output_dir / DATASETS["temperature"]["labels"]

    # Check if files already exist
    if inputs_path.exists() and labels_path.exists() and not force_download:
        step.info(f"Jena Climate CSV files already exist in {output_dir}")
        return str(inputs_path), str(labels_path)

    start_time = time.time()

    # URL from TensorFlow datasets
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = output_dir / "jena_climate_2009_2016.csv.zip"
    csv_path = output_dir / "jena_climate_2009_2016.csv"

    # Download if needed
    if not csv_path.exists() or force_download:
        if not zip_path.exists() or force_download:
            step.info(f"downloading from {url}")
            urllib.request.urlretrieve(url, zip_path)

        step.info("extracting zip file")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)

    # Load the dataset
    step.info("loading climate data")
    df = pd.read_csv(csv_path)

    # Extract temperature column (T is 2m temperature in Celsius)
    temp_column = 'T (degC)'
    if temp_column not in df.columns:
        # Try alternative column names
        for col in df.columns:
            if 'T' in col and 'deg' in col.lower():
                temp_column = col
                break
        else:
            step.error(f"temperature column not found; available: {list(df.columns)}")
            raise ValueError(f"Temperature column not found. Available: {list(df.columns)}")

    temp_series = df[temp_column]

    # Timestamps are used below to align the 2-hour decimation to clean
    # clock boundaries and to check that the raw rows are evenly spaced.
    # Decimation is purely positional (row N, N+12, N+24, ...), so it only
    # measures true 2-hour gaps if every row really is 10 minutes after the
    # last one; without the timestamps we'd have no way to tell.
    date_column = 'Date Time'
    timestamps = pd.to_datetime(
        df[date_column], format='%d.%m.%Y %H:%M:%S', errors='coerce'
    ) if date_column in df.columns else None

    # Interpolate missing values in temperature data
    missing_count = temp_series.isna().sum()
    if missing_count > 0:
        step.info(f"found {missing_count} missing temperature values, interpolating")
        temp_series = temp_series.interpolate(method='linear')
        # Forward fill any remaining NaN at the start, backward fill at the end
        temp_series = temp_series.ffill().bfill()

    temp_data_raw = temp_series.values.astype('float32')
    step.info(f"loaded {len(temp_data_raw)} raw temperature measurements (10-min resolution)")

    # The column labels this function produces (2h, 4h, ..., 168h / 170h,
    # ..., 192h) describe a series sampled every 2 hours, so the raw
    # 10-minute data is decimated to that resolution before windowing;
    # otherwise each "84-step" window would span 14 real hours instead of
    # the 168 hours the column names claim.
    # 2 hours / 10 minutes = 12, so every 12th raw reading is kept.
    RESAMPLE_STEP = 12  # 12 * 10min = 2h

    # Decimation assumes every raw row is exactly 10 minutes after the
    # last one; flag it if that's not true, since a violation would make
    # some windows span slightly more or less than 2 real hours per step
    # without any other visible symptom.
    if timestamps is not None:
        gaps = timestamps.diff().dropna()
        irregular = int((gaps != pd.Timedelta(minutes=10)).sum())
        if irregular:
            step.warn(
                f"{irregular} raw timestamp gaps are not exactly 10 minutes apart\n"
                f"{' '*10}2-hour decimation is positional, so windows spanning these points may drift\n"
                f"{' '*10}but these irregular gaps are minimal and safe to ignore for this dataset"
            )

    # Start decimating from the first row that lands on a clean 2-hour
    # clock boundary (e.g. 00:00, 02:00, ...) rather than index 0, so the
    # resulting cadence isn't offset by whatever minute the first raw row
    # happens to be on.
    offset = 0
    if timestamps is not None:
        for i in range(min(RESAMPLE_STEP, len(timestamps))):
            ts = timestamps.iloc[i]
            if pd.notna(ts) and ts.minute == 0 and ts.hour % 2 == 0:
                offset = i
                break

    temp_data = temp_data_raw[offset::RESAMPLE_STEP]
    step.info(
        f"resampled to 2-hour resolution: {len(temp_data)} measurements "
        f"(every {RESAMPLE_STEP}th raw reading, starting at offset {offset})"
    )

    # Create column names matching the expected CSV format
    # Input: 2h, 4h, 6h, ..., 168h (84 timesteps)
    input_columns = [f"{2*(i+1)}h" for i in range(84)]
    # Label: 170h, 172h, ..., 192h (12 timesteps)
    label_columns = [f"{170+2*i}h" for i in range(12)]

    # Create sliding windows: 84 input timesteps, 12 output timesteps
    seq_length = 84
    pred_length = 12
    window_size = seq_length + pred_length

    total_available = len(temp_data) - window_size + 1
    step.info(f"available windows at 2-hour resolution: {total_available}")

    if target_samples is None:
        # No cap; use every available window.
        target_samples = total_available
    elif target_samples > total_available:
        step.error(
            f"not enough data to generate {target_samples} samples; "
            f"only {total_available} windows available"
        )
        raise ValueError(
            f"Not enough data to generate {target_samples} samples. "
            f"Only {total_available} windows available from the raw data."
        )

    step.info(f"creating sliding windows (input: {seq_length}, predict: {pred_length})")

    # Pre-allocate arrays for efficiency
    inputs_array = np.zeros((target_samples, seq_length), dtype='float32')
    labels_array = np.zeros((target_samples, pred_length), dtype='float32')

    if target_samples == total_available:
        # Use every consecutive window (no subsampling needed).
        for idx in range(target_samples):
            inputs_array[idx] = temp_data[idx:idx+seq_length]
            labels_array[idx] = temp_data[idx+seq_length:idx+window_size]
    else:
        # Use evenly distributed indices to sample target_samples windows
        indices = np.linspace(0, total_available - 1, num=target_samples, dtype=int)
        for idx, i in enumerate(indices):
            inputs_array[idx] = temp_data[i:i+seq_length]
            labels_array[idx] = temp_data[i+seq_length:i+window_size]

    step.info(f"generated {len(inputs_array)} input samples and {len(labels_array)} label samples")

    # Save inputs CSV
    inputs_df = pd.DataFrame(inputs_array, columns=input_columns)
    inputs_df.to_csv(inputs_path, index=False, header=False)
    step.info(f"saved inputs: {inputs_path} ({inputs_array.shape[0]} samples, {inputs_array.shape[1]} features)")

    # Save labels CSV
    labels_df = pd.DataFrame(labels_array, columns=label_columns)
    labels_df.to_csv(labels_path, index=False, header=False)
    step.info(f"saved labels: {labels_path} ({labels_array.shape[0]} samples, {labels_array.shape[1]} features)")

    elapsed = time.time() - start_time
    step.info(f"completed in {elapsed:.2f}s")

    return str(inputs_path), str(labels_path)


def generate_all_datasets(
    output_dir: str = "datasets",
    force_download: bool = False,
) -> dict:
    """Generate all datasets from raw sources, using ALL available samples
    for both (no caps).

    Opens two child steps under the currently active step: "Generating
    MNIST" and "Generating Temperature (Jena Climate)".

    Args:
        output_dir: Directory to save CSV files
        force_download: If True, re-download all datasets

    Returns:
        Dictionary with paths to all generated CSV files:
        {
            'mnist_inputs': str,
            'mnist_labels': str,
            'temperature_inputs': str,
            'temperature_labels': str
        }

    Raises:
        ImportError: If torchvision is not installed (via download_mnist_csv)
        ValueError: If data issues occur (via download_jena_climate_csv)
    """
    step = current()
    results = {}

    # Generate MNIST; all 70,000 samples (train + test)
    with step.child("Generating MNIST"):
        mnist_inputs, mnist_labels = download_mnist_csv(
            output_dir, num_samples=None, force_download=force_download
        )
        results['mnist_inputs'] = mnist_inputs
        results['mnist_labels'] = mnist_labels

    # Generate Temperature (Jena Climate); all windows at 2-hour
    # resolution (~34,951), not the raw 10-minute row count.
    with step.child("Generating Temperature (Jena Climate)"):
        temp_inputs, temp_labels = download_jena_climate_csv(
            output_dir, target_samples=None, force_download=force_download
        )
        results['temperature_inputs'] = temp_inputs
        results['temperature_labels'] = temp_labels

    return results


def get_dataset_paths(
    dataset_name: str,
    output_dir: str = "datasets",
    try_download: bool = True,
) -> Tuple[str, str]:
    """Get dataset paths, optionally downloading if not found.

    Args:
        dataset_name: Either 'mnist' or 'temperature'
        output_dir: Directory where CSV files are stored
        try_download: If True, attempt to download if files don't exist

    Returns:
        Tuple of (inputs_path, labels_path)

    Raises:
        FileNotFoundError: If files don't exist and try_download is False
    """
    step = current()
    output_dir = Path(output_dir)

    if dataset_name == 'mnist':
        inputs_path = output_dir / DATASETS["mnist"]["inputs"]
        labels_path = output_dir / DATASETS["mnist"]["labels"]

        if try_download and (not inputs_path.exists() or not labels_path.exists()):
            with step.child("Generating MNIST"):
                return download_mnist_csv(output_dir, num_samples=None)
    elif dataset_name == 'temperature':
        inputs_path = output_dir / DATASETS["temperature"]["inputs"]
        labels_path = output_dir / DATASETS["temperature"]["labels"]

        if try_download and (not inputs_path.exists() or not labels_path.exists()):
            with step.child("Generating Temperature (Jena Climate)"):
                return download_jena_climate_csv(output_dir, target_samples=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'temperature'.")

    if not inputs_path.exists() or not labels_path.exists():
        step.error(f"dataset files not found in {output_dir}")
        raise FileNotFoundError(
            f"Dataset files not found in {output_dir}. "
            f"Inputs: {inputs_path.exists()}, Labels: {labels_path.exists()}. "
            f"Set try_download=True to download automatically."
        )

    return str(inputs_path), str(labels_path)


def ensure_datasets_exist(
    dataset_dir: str = "datasets",
    auto_download: bool = True,
) -> dict:
    """Ensure dataset CSV files exist, downloading from raw sources if needed.

    If anything is missing, opens a "Downloading datasets" child step under
    the currently active step (which itself nests "Generating MNIST" /
    "Generating Temperature (Jena Climate)" via generate_all_datasets). If
    everything already exists, just logs one info line and returns.

    Args:
        dataset_dir: Directory where CSV files should be stored
        auto_download: If True, automatically download and generate if files don't exist

    Returns:
        Dictionary with paths to all dataset files:
        {
            'temperature_inputs': str,
            'temperature_labels': str,
            'mnist_inputs': str,
            'mnist_labels': str
        }
    """
    step = current()
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check which datasets are missing
    expected_files = [
        DATASETS["temperature"]["inputs"],
        DATASETS["temperature"]["labels"],
        DATASETS["mnist"]["inputs"],
        DATASETS["mnist"]["labels"],
    ]

    missing_files = [f for f in expected_files if not (dataset_dir / f).exists()]

    paths = {
        'temperature_inputs': str(dataset_dir / DATASETS["temperature"]["inputs"]),
        'temperature_labels': str(dataset_dir / DATASETS["temperature"]["labels"]),
        'mnist_inputs': str(dataset_dir / DATASETS["mnist"]["inputs"]),
        'mnist_labels': str(dataset_dir / DATASETS["mnist"]["labels"]),
    }

    if missing_files and auto_download:
        step.info(f"missing datasets: {missing_files}")
        with step.child("Downloading datasets") as dl_step:
            dl_step.info("no sample caps; downloading from raw sources")
            generate_all_datasets(dataset_dir)
        return paths
    elif missing_files:
        step.error(f"dataset files missing in {dataset_dir}: {missing_files}")
        raise FileNotFoundError(
            f"Dataset files missing in {dataset_dir}: {missing_files}. "
            f"Set auto_download=True to generate them from raw sources."
        )
    else:
        step.info("all dataset files found locally")
        return paths


def cleanup_intermediate_files(output_dir: str = "datasets") -> None:
    """Remove intermediate download/cache artifacts, keeping only the
    final CSV outputs (mnist_digit_inputs.csv, mnist_digit_labels.csv,
    temperature_inputs.csv, temperature_labels.csv).

    Removes:
      - datasets/.mnist_cache/          (raw torchvision MNIST download)
      - datasets/jena_climate_2009_2016.csv      (unzipped raw Jena data)
      - datasets/jena_climate_2009_2016.csv.zip  (downloaded zip)

    Args:
        output_dir: directory to clean up.
    """
    step = current()
    output_dir = Path(output_dir)

    mnist_cache = output_dir / ".mnist_cache"
    if mnist_cache.exists():
        shutil.rmtree(mnist_cache)
        step.info(f"removed {mnist_cache}")

    jena_csv = output_dir / "jena_climate_2009_2016.csv"
    if jena_csv.exists():
        jena_csv.unlink()
        step.info(f"removed {jena_csv}")

    jena_zip = output_dir / "jena_climate_2009_2016.csv.zip"
    if jena_zip.exists():
        jena_zip.unlink()
        step.info(f"removed {jena_zip}")
