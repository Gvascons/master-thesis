#!/usr/bin/env python3
"""CLI script to download all datasets from OpenML."""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_all, download_dataset
from src.utils.logging import setup_logging


@click.command()
@click.option("--dataset", "-d", default=None, help="Download a specific dataset (by name)")
@click.option("--data-dir", default=None, type=click.Path(), help="Override data directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(dataset, data_dir, verbose):
    """Download datasets from OpenML for the tabular benchmark."""
    setup_logging(level="DEBUG" if verbose else "INFO")

    data_path = Path(data_dir) if data_dir else None

    if dataset:
        download_dataset(dataset, data_path)
    else:
        download_all(data_path)

    click.echo("Done!")


if __name__ == "__main__":
    main()
