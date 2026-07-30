#!/usr/bin/env python3
"""
Dataset Generation Script

This script downloads and generates the required CSV datasets from authoritative
sources, eliminating the need for pre-processed local CSV files.

Usage:
    python generate_datasets.py                    # Generate all datasets
    python generate_datasets.py --mnist            # Generate only MNIST
    python generate_datasets.py --temperature      # Generate only Temperature
    python generate_datasets.py --force            # Force re-download
    python generate_datasets.py --output /path/to/dir  # Custom output directory
    python generate_datasets.py --keep-intermediate    # Keep raw download artifacts
"""

import argparse
from pathlib import Path

from src.richlog import Logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets from raw sources"
    )
    parser.add_argument(
        '--mnist',
        action='store_true',
        help='Generate only MNIST dataset'
    )
    parser.add_argument(
        '--temperature',
        action='store_true',
        help='Generate only Temperature dataset'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets',
        help='Output directory for CSV files (default: datasets)'
    )
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep raw download artifacts (.mnist_cache/, jena_climate_2009_2016.csv, .zip)'
    )
    args = parser.parse_args()

    # Import after argument parsing
    from src.data.generation import (
        download_mnist_csv,
        download_jena_climate_csv,
        generate_all_datasets,
        cleanup_intermediate_files,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is a standalone CLI entrypoint (separate from main.py's run), so
    # it gets its own Logger and its own logs/run_<timestamp>.log file.
    with Logger(log_dir="logs") as log:
        with log.step("Generate datasets") as step:
            if args.mnist and not args.temperature:
                with step.child("Generating MNIST"):
                    download_mnist_csv(output_dir, force_download=args.force)
            elif args.temperature and not args.mnist:
                with step.child("Generating Temperature (Jena Climate)"):
                    download_jena_climate_csv(output_dir, force_download=args.force)
            else:
                generate_all_datasets(output_dir, force_download=args.force)

            if not args.keep_intermediate:
                with step.child("Cleaning up intermediate files"):
                    cleanup_intermediate_files(output_dir)

            step.info(f"files saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
