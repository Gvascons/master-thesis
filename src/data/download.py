"""Download and cache datasets from OpenML."""

import logging
import shutil
import time
from pathlib import Path

import openml
import pandas as pd

from src.utils.config import load_datasets_config, load_experiment_config

logger = logging.getLogger("tabular_benchmark")


def download_dataset(name: str, data_dir: Path | None = None, max_retries: int = 3) -> Path:
    """Download a single dataset from OpenML and save as parquet.

    Retries on network failures. Returns the path to the saved parquet file.
    """
    datasets_cfg = load_datasets_config()
    exp_cfg = load_experiment_config()

    # Find dataset in classification or regression
    ds_cfg = None
    for task_group in ("classification", "regression"):
        if name in datasets_cfg[task_group]:
            ds_cfg = datasets_cfg[task_group][name]
            break
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {name}. Check configs/datasets.yaml")

    if data_dir is None:
        data_dir = Path(exp_cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / f"{name}.parquet"
    if out_path.exists():
        logger.info(f"Dataset '{name}' already cached at {out_path}")
        return out_path

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading '{name}' (OpenML ID={ds_cfg.openml_id})... (attempt {attempt}/{max_retries})")
            # Clear OpenML cache for this dataset to avoid partial downloads
            if attempt > 1:
                _clear_openml_cache(ds_cfg.openml_id)

            dataset = openml.datasets.get_dataset(
                ds_cfg.openml_id,
                version=ds_cfg.get("version", None),
                download_data=True,
                download_qualities=False,
                download_features_meta_data=True,
            )
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
            )
            df = pd.DataFrame(X, columns=attribute_names)
            df["target"] = y

            df.to_parquet(out_path, index=False)
            logger.info(f"Saved '{name}' -> {out_path}  ({len(df)} rows, {len(df.columns)-1} features)")
            return out_path

        except (ConnectionError, OSError, TimeoutError) as e:
            if attempt == max_retries:
                raise
            wait = 5 * attempt
            logger.warning(f"Download failed for '{name}': {e}. Retrying in {wait}s...")
            time.sleep(wait)

    return out_path  # unreachable, but satisfies type checker


def _clear_openml_cache(dataset_id: int) -> None:
    """Remove cached files for a dataset so retries start fresh."""
    cache_dir = Path(openml.config.get_cache_directory()) / "datasets" / str(dataset_id)
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.debug(f"Cleared OpenML cache for dataset {dataset_id}")


def download_all(data_dir: Path | None = None) -> dict[str, Path]:
    """Download all datasets defined in the config. Continues past individual failures."""
    datasets_cfg = load_datasets_config()
    paths = {}
    failed = []
    for task_group in ("classification", "regression"):
        for name in datasets_cfg[task_group]:
            try:
                paths[name] = download_dataset(name, data_dir)
            except Exception as e:
                logger.error(f"Failed to download '{name}': {e}")
                failed.append(name)
    if failed:
        logger.warning(f"Failed datasets: {failed}. Re-run to retry.")
    return paths
