#!/usr/bin/env python3
"""CLI: Orchestrate the full experimental pipeline.

Runs all model+dataset combinations, then aggregates results.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --gpu 0
    python scripts/run_all.py --models xgboost lightgbm --datasets adult wine_quality
    python scripts/run_all.py --force   # re-run even if results exist
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.registry import get_all_dataset_names
from src.models.factory import list_models
from src.utils.config import load_experiment_config
from src.utils.environment import capture_environment
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed
from src.utils.timer import Timer


@click.command()
@click.option("--models", "-m", multiple=True, default=None, help="Models to run (default: all)")
@click.option("--datasets", "-d", multiple=True, default=None, help="Datasets to run (default: all)")
@click.option("--gpu", "-g", default=None, type=int, help="GPU device ID")
@click.option("--results-dir", default="results", type=click.Path())
@click.option("--seed", default=42, type=int)
@click.option("--force", is_flag=True, help="Re-run experiments even if results already exist")
@click.option("--in-process", is_flag=True, help="Run experiments in-process (default: isolated subprocess each)")
@click.option("--verbose", "-v", is_flag=True)
def main(models, datasets, gpu, results_dir, seed, force, in_process, verbose):
    """Run the full experimental pipeline."""
    setup_logging(level="DEBUG" if verbose else "INFO",
                  log_file=Path(results_dir) / "logs" / "run_all.log")
    logger = get_logger()

    # Capture and save environment info
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    env_info = capture_environment()
    env_path = results_path / "environment.json"
    env_path.write_text(json.dumps(env_info, indent=2, default=str))
    logger.info(f"Environment info saved to {env_path}")

    exp_cfg = load_experiment_config()
    set_seed(seed)

    model_list = list(models) if models else list_models()
    dataset_list = list(datasets) if datasets else get_all_dataset_names()

    total = len(model_list) * len(dataset_list)
    logger.info(f"Running {len(model_list)} models x {len(dataset_list)} datasets = {total} experiments")

    # By default each experiment runs in its own subprocess so that a CUDA
    # out-of-memory error (which can poison the CUDA context irrecoverably)
    # cannot cascade to later experiments — the process exits, its GPU memory
    # is fully reclaimed by the OS, and the next experiment starts clean.
    train_script = Path(__file__).resolve().parent / "train.py"
    train_single = None
    if in_process:
        from scripts.train import train_single  # noqa: F401

    raw_dir = results_path / "raw"
    completed = 0
    skipped = 0
    failed = 0

    overall_timer = Timer()
    with overall_timer:
        for ds_name in dataset_list:
            for model_name in model_list:
                # Resume capability: skip if result already exists
                result_file = raw_dir / f"{model_name}_{ds_name}.json"
                if result_file.exists() and not force:
                    skipped += 1
                    logger.info(f"Skipping {model_name} x {ds_name} (already completed)")
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"[{completed+failed+skipped+1}/{total}] {model_name} x {ds_name}")
                logger.info(f"{'='*60}")

                try:
                    timer = Timer()
                    with timer:
                        if in_process:
                            train_single(model_name, ds_name, results_path, gpu=gpu, seed=seed)
                        else:
                            cmd = [
                                sys.executable, str(train_script),
                                "--model", model_name,
                                "--dataset", ds_name,
                                "--results-dir", str(results_path),
                                "--seed", str(seed),
                            ]
                            if gpu is not None:
                                cmd += ["--gpu", str(gpu)]
                            if verbose:
                                cmd += ["--verbose"]
                            subprocess.run(cmd, check=True, env=os.environ.copy())
                    # In subprocess mode, success is confirmed by the result file
                    if in_process or result_file.exists():
                        completed += 1
                        logger.info(f"Completed in {timer.result.elapsed:.1f}s")
                    else:
                        failed += 1
                        logger.error(f"FAILED: {model_name} x {ds_name}: no result file produced")
                except subprocess.CalledProcessError as e:
                    failed += 1
                    logger.error(f"FAILED: {model_name} x {ds_name}: subprocess exited with code {e.returncode}")
                except Exception as e:
                    failed += 1
                    logger.error(f"FAILED: {model_name} x {ds_name}: {e}", exc_info=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete: {completed} succeeded, {skipped} skipped, {failed} failed")
    logger.info(f"Total time: {overall_timer.result.elapsed:.1f}s")

    # Aggregate results
    if completed > 0 or skipped > 0:
        logger.info("Aggregating results...")
        from scripts.evaluate import aggregate_results
        aggregate_results(results_path)


if __name__ == "__main__":
    main()
