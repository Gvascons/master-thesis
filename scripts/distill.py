#!/usr/bin/env python3
"""Distillation harness (AI-2): distributional regression distillation.

Design: docs/desenho-experimental-destilacao.md. Pipeline per dataset:

  Stage "oof"      Teacher (TabPFN v2.5) labels the training pool
                   out-of-fold (K=5): predictive mean + quantile grid per
                   row. Also fits the teacher on the full pool and scores
                   the fixed benchmark hold-out (the accuracy anchor).
                   Cached to results/distillation/oof/<teacher>_<ds>.parquet
                   and .../teacher_eval.csv — paid once, reused by all
                   student variants.

  Stage "students" Students trained on the cached targets and evaluated on
                   the same hold-out. Variants (design §2):
                     xgb_point/hard    XGBoost on true y            (control)
                     xgb_point/soft    XGBoost on teacher OOF mean  (distill)
                     xgb_point/mixed   λ·soft + (1−λ)·hard, λ=0.8
                     xgb_quant/hard    multi-quantile pinball on y  (control)
                     xgb_quant/soft    multi-output MSE on teacher quantile
                                       curves (distills the distribution)
                   Student HPs = the benchmark's tuned XGBoost best_params
                   (controlled comparison: same capacity, only the target
                   changes; re-tuning per regime is a later ablation).

Metrics per (dataset, student, target, seed): RMSE, MAE, CRPS (pinball
integral over the quantile grid), PICP80/90 + interval widths, retention of
the teacher−baseline RMSE gap. Appended incrementally to
results/distillation/distill.csv; reruns skip completed rows.

Usage:
    uv run python scripts/distill.py --stage oof --datasets wine_quality \
        --device cpu            # smoke test without touching the busy GPU
    uv run python scripts/distill.py --stage oof                # full (GPU)
    uv run python scripts/distill.py --stage students           # CPU
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.registry import get_holdout_split, load_dataset
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

SEED = 42
TEST_SIZE = 0.2
OOF_FOLDS = 5
QUANTILES = [round(q, 2) for q in np.arange(0.05, 0.96, 0.05)]  # 19 levels
LAMBDA_MIXED = 0.8

OUT_DIR = REPO / "results" / "distillation"
OOF_DIR = OUT_DIR / "oof"
TEACHER_EVAL = OUT_DIR / "teacher_eval.csv"
STUDENT_CSV = OUT_DIR / "distill.csv"

REG_DATASETS = ["wine_quality", "california_housing", "superconduct",
                "kin8nm", "year_prediction"]

STUDENT_HEADER = [
    "dataset", "teacher", "student", "target", "seed", "n_train", "n_test",
    "rmse", "mae", "crps", "picp80", "picp90", "width80", "width90",
    "retention_rmse", "fit_time_s",
]


# --------------------------------------------------------------------------- #
# shared: pool/hold-out identical to the benchmark
# --------------------------------------------------------------------------- #
def prep_pool(dataset_name, exp_cfg, cap=None):
    X, y, info = load_dataset(dataset_name)
    max_samples = cap or getattr(exp_cfg, "max_dataset_samples", None)
    if max_samples and len(X) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, random_state=SEED)
    X_pool, y_pool, X_test, y_test = get_holdout_split(
        X, y, info, seed=SEED, test_size=TEST_SIZE)
    return X_pool, y_pool, X_test, y_test, info


def artifact_paths(cap):
    """Smoke runs (--cap) write to separate files so they never pollute the
    real caches/CSVs."""
    suffix = f"_cap{cap}" if cap else ""
    return (OUT_DIR / f"teacher_eval{suffix}.csv",
            OUT_DIR / f"distill{suffix}.csv", suffix)


def ordinal_encode(X_pool, X_test):
    """TabPFN-family preprocessing: ordinal-encode categoricals, keep numerics."""
    Xp = X_pool.copy()
    Xt = X_test.copy()
    for col in Xp.columns:
        if Xp[col].dtype == "object" or str(Xp[col].dtype) == "category":
            cats = pd.Categorical(Xp[col]).categories
            Xp[col] = pd.Categorical(Xp[col], categories=cats).codes
            Xt[col] = pd.Categorical(Xt[col], categories=cats).codes
    return Xp.to_numpy(dtype=np.float32), Xt.to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pinball(y, q_pred, tau):
    diff = y - q_pred
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))


def crps_from_quantiles(y, Q):
    """CRPS approximated by the pinball-loss integral over the quantile grid.

    Q: (n, K) matrix of predicted quantiles at levels QUANTILES.
    """
    return float(2.0 * np.mean([pinball(y, Q[:, k], tau)
                                for k, tau in enumerate(QUANTILES)]))


def interval_metrics(y, Q):
    """PICP and mean width for the 80% [q10,q90] and 90% [q05,q95] intervals."""
    i = {q: QUANTILES.index(q) for q in (0.05, 0.10, 0.90, 0.95)}
    out = {}
    for name, lo, hi in (("80", 0.10, 0.90), ("90", 0.05, 0.95)):
        lo_v, hi_v = Q[:, i[lo]], Q[:, i[hi]]
        out[f"picp{name}"] = float(np.mean((y >= lo_v) & (y <= hi_v)))
        out[f"width{name}"] = float(np.mean(hi_v - lo_v))
    return out


def sort_quantiles(Q):
    """Fix quantile crossing by row-wise sorting (design §2, documented)."""
    return np.sort(Q, axis=1)


# --------------------------------------------------------------------------- #
# stage: oof — teacher labels the pool out-of-fold + anchors on the hold-out
# --------------------------------------------------------------------------- #
TEACHER_MAX_ROWS = 50_000  # TabPFN v2.5 native limit; mirrors tabpfn_max_samples


def make_teacher(device, teacher="tabpfn"):
    if teacher == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device=device, n_estimators=8, random_state=SEED)
    # TabFM: point-only teacher (mean of the ensemble); n_estimators=8 and the
    # 50K row cap mirror our benchmark TabFM wrapper for comparability.
    import torch
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
    dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    backbone = tabfm_v1_0_0_pytorch.load("regression", device=dev)
    return TabFMRegressor(model=backbone, n_estimators=8,
                          max_num_rows=TEACHER_MAX_ROWS, random_state=SEED)


def fit_teacher(model, X, y):
    """Fit respecting the teacher's 50K context limit (same subsampling
    policy as the benchmark's TabPFN wrapper, for comparability)."""
    if len(y) > TEACHER_MAX_ROWS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y), size=TEACHER_MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    model.fit(X, y)
    return model

def teacher_predict(model, X, batch=2000, teacher="tabpfn"):
    """Mean + quantile grid, chunked over rows: predicting a large fold in
    one pass against a 50K context OOMs the 16GB GPU (predictions are
    independent, so chunking bounds peak memory without changing results —
    same rationale as the benchmark's TabPFN wrapper)."""
    means, Qs = [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        if teacher == "tabpfn":
            means.append(np.asarray(model.predict(xb, output_type="mean")))
            qs = model.predict(xb, output_type="quantiles", quantiles=QUANTILES)
            Qs.append(np.column_stack([np.asarray(q) for q in qs]))
        else:  # tabfm: point-only
            means.append(np.asarray(model.predict(xb)))
    if not Qs:
        return np.concatenate(means), None
    return np.concatenate(means), sort_quantiles(np.vstack(Qs))


def free_teacher(model):
    """Release the teacher's GPU context between folds."""
    del model
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_oof(datasets, device, cap=None, teacher="tabpfn"):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    exp_cfg = load_experiment_config()
    for ds in datasets:
        teacher_eval, _, suffix = artifact_paths(cap)
        oof_path = OOF_DIR / f"{teacher}_{ds}{suffix}.parquet"
        if oof_path.exists():
            print(f"[skip] OOF ja existe: {oof_path.name}", flush=True)
            continue
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg, cap)
        assert info.task_type == "regression", f"{ds} nao e regressao"
        Xp, Xt = ordinal_encode(X_pool, X_test)
        n = len(y_pool)
        print(f"=== OOF {teacher}/{ds}: pool={n} test={len(y_test)} "
              f"device={device} ===", flush=True)

        mean_oof = np.full(n, np.nan)
        Q_oof = np.full((n, len(QUANTILES)), np.nan)
        kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)
        t0 = time.perf_counter()
        for fi, (tr, ho) in enumerate(kf.split(Xp)):
            model = make_teacher(device, teacher)
            fit_teacher(model, Xp[tr], y_pool[tr])
            m, Q = teacher_predict(model, Xp[ho], teacher=teacher)
            mean_oof[ho] = m
            if Q is not None:
                Q_oof[ho] = Q
            free_teacher(model)
            print(f"  fold {fi+1}/{OOF_FOLDS} rotulado "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        assert not np.isnan(mean_oof).any()

        if np.isnan(Q_oof).all():   # point-only teacher: no quantile columns
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(Q_oof, columns=[f"q{int(t*100):02d}" for t in QUANTILES])
        df.insert(0, "mean", mean_oof)
        df.insert(0, "y", y_pool)
        df.to_parquet(oof_path)
        print(f"  OOF salvo: {oof_path.name}", flush=True)

        # teacher anchor on the hold-out (fit on the FULL pool, as deployed)
        model = make_teacher(device, teacher)
        fit_teacher(model, Xp, y_pool)
        m, Q = teacher_predict(model, Xt, teacher=teacher)
        free_teacher(model)
        rmse = float(np.sqrt(np.mean((y_test - m) ** 2)))
        row = {"dataset": ds, "teacher": teacher, "rmse": round(rmse, 6),
               "mae": round(float(np.mean(np.abs(y_test - m))), 6),
               "crps": round(crps_from_quantiles(y_test, Q), 6) if Q is not None else "",
               **({k: round(v, 6) for k, v in interval_metrics(y_test, Q).items()}
                  if Q is not None else
                  {k: "" for k in ("picp80", "width80", "picp90", "width90")}),
               "n_test": len(y_test)}
        hdr = list(row.keys())
        write_hdr = not teacher_eval.exists()
        with open(teacher_eval, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if write_hdr:
                w.writeheader()
            w.writerow(row)
        print(f"  teacher anchor: rmse={rmse:.4f} crps={row['crps'] or '-'}", flush=True)


# --------------------------------------------------------------------------- #
# stage: students
# --------------------------------------------------------------------------- #
def load_xgb_params(ds):
    jpath = REPO / "results" / "raw" / f"xgboost_{ds}.json"
    params = json.loads(jpath.read_text()).get("best_params", {}) or {}
    params.pop("cat_feature_indices", None)
    return params


def already_done(student_csv):
    done = set()
    if student_csv.exists():
        with open(student_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["teacher"], r["dataset"], r["student"], r["target"], int(r["seed"])))
    return done


def append_student_row(row, student_csv):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_hdr = not student_csv.exists()
    with open(student_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STUDENT_HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def fit_xgb_point(Xtr, target, params, seed):
    from xgboost import XGBRegressor
    m = XGBRegressor(**params, random_state=seed, n_jobs=-1)
    m.fit(Xtr, target)
    return m


def fit_xgb_quant_hard(Xtr, y, params, seed):
    """Native multi-quantile pinball regression on true labels."""
    from xgboost import XGBRegressor
    p = dict(params)
    p.pop("objective", None)
    m = XGBRegressor(**p, objective="reg:quantileerror",
                     quantile_alpha=np.array(QUANTILES),
                     random_state=seed, n_jobs=-1)
    m.fit(Xtr, y)
    return m


def fit_xgb_quant_soft(Xtr, Q_targets, params, seed):
    """Multi-output MSE on the teacher's quantile curves (distills the
    distribution as 19 regression surfaces; crossing fixed post-hoc)."""
    from xgboost import XGBRegressor
    m = XGBRegressor(**params, random_state=seed, n_jobs=-1,
                     multi_strategy="one_output_per_tree")
    m.fit(Xtr, Q_targets)
    return m


def run_students(datasets, seeds, cap=None, teacher="tabpfn"):
    exp_cfg = load_experiment_config()
    teacher_eval, student_csv, suffix = artifact_paths(cap)
    done = already_done(student_csv)
    baseline = pd.read_csv(REPO / "results" / "aggregated" / "test_results.csv")

    for ds in datasets:
        oof_path = OOF_DIR / f"{teacher}_{ds}{suffix}.parquet"
        if not oof_path.exists():
            print(f"[skip] sem OOF para {ds} — rode --stage oof antes", flush=True)
            continue
        oof = pd.read_parquet(oof_path)
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg, cap)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        assert len(oof) == len(y_pool), "OOF desalinhado com o pool"
        params = load_xgb_params(ds)
        qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

        b = baseline[(baseline.dataset == ds) & (baseline.model == "xgboost")]
        rmse_baseline = float(b.rmse.iloc[0])
        t_eval = pd.read_csv(teacher_eval)
        t_row = t_eval[(t_eval.dataset == ds) & (t_eval.teacher == teacher)]
        rmse_teacher = float(t_row.rmse.iloc[0]) if len(t_row) else np.nan

        targets = {
            "hard": y_pool,
            "soft": oof["mean"].to_numpy(),
            "mixed": LAMBDA_MIXED * oof["mean"].to_numpy() + (1 - LAMBDA_MIXED) * y_pool,
        }
        print(f"=== students {ds}: baseline_rmse={rmse_baseline:.4f} "
              f"teacher_rmse={rmse_teacher:.4f} ===", flush=True)

        has_quantiles = all(c in oof.columns for c in qcols)
        for seed in seeds:
            variants = (
                [("xgb_point", tname, lambda tv=tvec: fit_xgb_point(Xp, tv, params, seed))
                 for tname, tvec in targets.items()]
                + [("xgb_quant", "hard", lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed))]
                + ([("xgb_quant", "soft", lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed))]
                   if has_quantiles else [])
            )
            for student, tname, fit_fn in variants:
                if (teacher, ds, student, tname, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                ft = time.perf_counter() - t0
                pred = model.predict(Xt)

                if student == "xgb_point":
                    point = pred
                    Q = None
                else:
                    Q = sort_quantiles(np.asarray(pred))
                    point = Q[:, QUANTILES.index(0.50)]  # median as point pred

                rmse = float(np.sqrt(np.mean((y_test - point) ** 2)))
                row = {
                    "dataset": ds, "teacher": teacher, "student": student,
                    "target": tname, "seed": seed, "n_train": len(y_pool),
                    "n_test": len(y_test), "rmse": round(rmse, 6),
                    "mae": round(float(np.mean(np.abs(y_test - point))), 6),
                    "crps": round(crps_from_quantiles(y_test, Q), 6) if Q is not None else "",
                    "fit_time_s": round(ft, 2),
                }
                im = interval_metrics(y_test, Q) if Q is not None else {}
                for k in ("picp80", "picp90", "width80", "width90"):
                    row[k] = round(im[k], 6) if im else ""
                gap = rmse_baseline - rmse_teacher
                row["retention_rmse"] = (
                    round((rmse_baseline - rmse) / gap, 4) if gap > 1e-12 else "")
                append_student_row(row, student_csv)
                print(f"  [{student:9s}/{tname:5s} seed={seed}] rmse={rmse:.4f} "
                      f"crps={row['crps'] or '-'} ret={row['retention_rmse']} "
                      f"fit={ft:.1f}s", flush=True)

    print("\nSTUDENTS DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oof", "students"], required=True)
    ap.add_argument("--datasets", default=",".join(REG_DATASETS))
    ap.add_argument("--device", default="auto", help="tabpfn device (cpu para smoke test)")
    ap.add_argument("--seeds", default=3, type=int)
    ap.add_argument("--cap", default=None, type=int, help="subamostra o dataset (smoke test; artefatos separados)")
    ap.add_argument("--teacher", default="tabpfn", choices=["tabpfn", "tabfm"])
    args = ap.parse_args()
    datasets = args.datasets.split(",")
    if args.stage == "oof":
        run_oof(datasets, args.device, cap=args.cap, teacher=args.teacher)
    else:
        run_students(datasets, list(range(args.seeds)), cap=args.cap, teacher=args.teacher)


if __name__ == "__main__":
    main()
