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
    
    print("="*80)
    print("Dataset Generation from Raw Sources")
    print("="*80)
    print()
    
    if args.mnist and not args.temperature:
        print("Generating MNIST dataset...")
        download_mnist_csv(output_dir, force_download=args.force)
        print("\nMNIST generation complete!")
    elif args.temperature and not args.mnist:
        print("Generating Temperature dataset (Jena Climate)...")
        download_jena_climate_csv(output_dir, force_download=args.force)
        print("\nTemperature generation complete!")
    else:
        print("Generating all datasets...")
        generate_all_datasets(output_dir, force_download=args.force)
    
    if not args.keep_intermediate:
        print("\nCleaning up intermediate download artifacts...")
        cleanup_intermediate_files(output_dir)
    
    print("\n" + "="*80)
    print("Dataset generation complete!")
    print(f"Files saved to: {output_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()