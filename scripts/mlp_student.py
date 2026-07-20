#!/usr/bin/env python3
"""MLP-quantile student (AI-2 phase 4): does the student family matter?

Same experimental cell as the XGBoost quantile students, with an MLP body:
the benchmark MLP architecture (per-dataset tuned params from
results/raw/mlp_<ds>.json) with a 19-output quantile head.

Variants (mirroring the XGB pair):
  hard  pinball loss on true y (classic quantile regression control)
  soft  MSE on the teacher's OOF quantile curves (distillation)

Early stopping on a 10% validation split with the same objective; standard
DL protocol (AdamW, batch 256, patience 20, max 200 epochs). Metrics on the
benchmark hold-out: RMSE (median as point), CRPS, PICP80. 3 seeds.
Output: results/distillation/mlp_students.csv (resumable).

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/mlp_student.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, OOF_DIR, crps_from_quantiles,
                     interval_metrics, ordinal_encode, prep_pool,
                     sort_quantiles)
from src.models.mlp_model import _MLP
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "distillation" / "mlp_students.csv"
DATASETS = ["wine_quality", "california_housing", "superconduct",
            "diamonds", "year_prediction"]
SEEDS = [0, 1, 2]
HEADER = ["dataset", "student", "target", "seed", "rmse", "crps", "picp80",
          "fit_time_s"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAUS = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["target"], int(r["seed"])))
    return done


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def pinball_loss(pred, y):
    """pred: (B, K) quantiles; y: (B,). Mean pinball over the grid."""
    diff = y.unsqueeze(1) - pred
    return torch.mean(torch.maximum(TAUS * diff, (TAUS - 1) * diff))


def load_mlp_params(ds):
    p = json.loads((REPO / "results" / "raw" / f"mlp_{ds}.json").read_text())
    return p.get("best_params", {}) or {}


def zscale(Xtr, *others):
    """Standardize numerics on the train split (MLP-family preprocessing)."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return tuple((X - mu) / sd for X in (Xtr, *others))


def fit_eval(ds, target_name, Xp, y_pool, Q_targets, Xt, y_test, params, seed):
    set_seed(seed)
    idx = np.arange(len(y_pool))
    tr, va = train_test_split(idx, test_size=0.1, random_state=seed)
    Xtr, Xva, Xte = zscale(Xp[tr], Xp[va], Xt)

    net = _MLP(n_features=Xp.shape[1],
               d_hidden=params.get("d_hidden", 256),
               n_blocks=params.get("n_blocks", 2),
               dropout=params.get("dropout", 0.1),
               d_out=len(QUANTILES)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(),
                            lr=params.get("learning_rate", 1e-4),
                            weight_decay=params.get("weight_decay", 1e-5))

    if target_name == "hard":
        ytr = torch.tensor(y_pool[tr], dtype=torch.float32)
        yva = torch.tensor(y_pool[va], dtype=torch.float32)
        loss_fn = pinball_loss
    else:  # soft: MSE nas curvas de quantis do teacher (simetrico ao XGB)
        ytr = torch.tensor(Q_targets[tr], dtype=torch.float32)
        yva = torch.tensor(Q_targets[va], dtype=torch.float32)
        loss_fn = lambda pred, y: nn.functional.mse_loss(pred, y)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), ytr),
        batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = yva.to(DEVICE)

    best, best_state, patience = float("inf"), None, 0
    t0 = time.perf_counter()
    for epoch in range(200):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xva_t), yva_t))
        if vl < best:
            best, patience = vl, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    ft = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(DEVICE)

    net.eval()
    preds = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        for i in range(0, len(Xte_t), 4096):
            preds.append(net(Xte_t[i:i+4096].to(DEVICE)).cpu().numpy())
    Q = sort_quantiles(np.vstack(preds))
    point = Q[:, QUANTILES.index(0.50)]
    return {
        "rmse": round(float(np.sqrt(np.mean((y_test - point) ** 2))), 6),
        "crps": round(crps_from_quantiles(y_test, Q), 6),
        "picp80": round(interval_metrics(y_test, Q)["picp80"], 4),
        "fit_time_s": round(ft, 2),
    }


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    for ds in DATASETS:
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
        params = load_mlp_params(ds)
        print(f"=== {ds}: pool={len(y_pool)} ===", flush=True)

        for seed in SEEDS:
            for target in ("hard", "soft"):
                if (ds, target, seed) in done:
                    continue
                m = fit_eval(ds, target, Xp, y_pool, oof[qcols].to_numpy(),
                             Xt, y_test, params, seed)
                append({"dataset": ds, "student": "mlp_quant",
                        "target": target, "seed": seed, **m})
                print(f"  [mlp_quant/{target} seed={seed}] rmse={m['rmse']} "
                      f"crps={m['crps']} picp80={m['picp80']} "
                      f"fit={m['fit_time_s']}s", flush=True)

    print("\nMLP STUDENTS DONE", flush=True)


if __name__ == "__main__":
    main()
