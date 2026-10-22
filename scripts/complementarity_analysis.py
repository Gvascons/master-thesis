#!/usr/bin/env python3
"""E7 — geração das predições por instância no hold-out (pré-registro:
docs/desenho-analise-complementaridade.md, 22/10/2026).

Reproduz o modelo FINAL do benchmark (scripts/train.py, Fase 3) para
xgboost/tabpfn/tabfm usando os best_params versionados em results/raw,
e salva as predições por instância do hold-out em
results/complementarity/preds_<modelo>_<dataset>.parquet, com gate de
sanidade contra as métricas salvas (sanity_check.csv).

Uso: uv run python scripts/complementarity_analysis.py [--models m ...]
     [--datasets d ...]   (resumível: pula parquets existentes)
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import get_model_family
from src.utils.config import load_experiment_config
from src.utils.gpu import create_fit_with_retry, free_gpu_memory
from src.utils.reproducibility import set_seed

OUT = REPO / "results/complementarity"
MODELS = ["xgboost", "tabpfn", "tabfm"]
PRIMARY = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
TOL = 1e-3  # gate pré-registrado na métrica primária


def run_one(model_name: str, ds_name: str, seed: int = 42) -> dict | None:
    raw = REPO / f"results/raw/{model_name}_{ds_name}.json"
    if not raw.exists():  # célula estrutural ausente (ex.: FMs x helena)
        print(f"  sem célula no benchmark: {model_name} x {ds_name} — pulado")
        return None
    stored = json.loads(raw.read_text())
    best_params = stored["best_params"]

    exp_cfg = load_experiment_config()
    set_seed(seed)
    X, y, info = load_dataset(ds_name)

    max_samples = getattr(exp_cfg, "max_dataset_samples", None)
    if max_samples and len(X) > max_samples:
        stratify = y if info.task_type in ("binary", "multiclass") else None
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, random_state=seed, stratify=stratify,
        )

    X_pool, y_pool, X_test, y_test = get_holdout_split(
        X, y, info, seed=seed, test_size=exp_cfg.test_size,
    )

    # Fase 3 do train.py, idêntica: refit em 90/10 do pool
    stratify = y_pool if info.task_type != "regression" else None
    X_ftr, X_fva, y_ftr, y_fva = train_test_split(
        X_pool, y_pool, test_size=0.1, random_state=seed, stratify=stratify,
    )
    preprocess_fn = get_preprocessor(get_model_family(model_name))
    prep = preprocess_fn(X_ftr, info, X_val=X_fva, X_test=X_test)
    kwargs = dict(**best_params)
    if prep.cat_feature_indices is not None:
        kwargs["cat_feature_indices"] = prep.cat_feature_indices

    set_seed(seed)
    model = create_fit_with_retry(
        model_name, info.task_type, info.n_classes, seed, kwargs,
        prep.X_train, y_ftr, prep.X_val, y_fva,
    )

    metrics = compute_all_metrics(model, prep.X_test, y_test, info.task_type)
    y_pred = model.predict(prep.X_test)

    df = pd.DataFrame({"row": np.asarray(X_test.index), "y_true": y_test})
    if info.task_type in ("binary", "multiclass"):
        proba = model.predict_proba(prep.X_test)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        df["y_pred"] = y_pred
        for k in range(proba.shape[1]):
            df[f"p{k}"] = proba[:, k]
    else:
        df["y_pred"] = y_pred
        df["abs_err"] = np.abs(y_test - y_pred)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"preds_{model_name}_{ds_name}.parquet", index=False)

    pm = PRIMARY[info.task_type]
    rec, sto = metrics.get(pm), stored["test_metrics"].get(pm)
    delta = abs(rec - sto) if rec is not None and sto is not None else np.nan
    status = "OK" if delta <= TOL else "DIVERGE"
    print(f"  {model_name} x {ds_name}: {pm} recomputado {rec:.6f} vs salvo "
          f"{sto:.6f} (|d|={delta:.2e}) -> {status}")

    del model
    free_gpu_memory()
    return {"model": model_name, "dataset": ds_name, "task_type": info.task_type,
            "metric": pm, "recomputed": rec, "stored": sto, "abs_delta": delta,
            "gate": status, "n_test": len(y_test), "n_pool": len(y_pool)}


@click.command()
@click.option("--models", "-m", multiple=True, default=MODELS)
@click.option("--datasets", "-d", multiple=True, default=None)
def main(models, datasets):
    import yaml
    cfg = yaml.safe_load((REPO / "configs/datasets.yaml").read_text())
    all_ds = [n for sec in ("classification", "regression")
              for n in cfg.get(sec, {})]
    ds_list = list(datasets) if datasets else all_ds

    sanity_path = OUT / "sanity_check.csv"
    rows = (pd.read_csv(sanity_path).to_dict("records")
            if sanity_path.exists() else [])
    done = {(r["model"], r["dataset"]) for r in rows}

    for ds in ds_list:
        print(f"== {ds} ==")
        for m in models:
            if (m, ds) in done and (OUT / f"preds_{m}_{ds}.parquet").exists():
                print(f"  {m} x {ds}: ja existe — pulado")
                continue
            try:
                rec = run_one(m, ds)
            except Exception as e:
                print(f"  FALHOU {m} x {ds}: {e}")
                rec = {"model": m, "dataset": ds, "gate": f"ERRO: {e}"}
            if rec:
                rows = [r for r in rows
                        if not (r["model"] == m and r["dataset"] == ds)]
                rows.append(rec)
                OUT.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(sanity_path, index=False)
    print("concluido.")


if __name__ == "__main__":
    main()
