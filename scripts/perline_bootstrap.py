"""Bootstrap por linhas do teste, por conjunto (QP3) — cumprimento da análise
prometida no pré-registro de 15/07 (eb0f223: "intervalos por bootstrap de
linhas") e exigida pela auditoria externa de 23/11 (correção obrigatória 6).

Para cada um dos 20 conjuntos da QP3, re-treina os dois braços quantílicos
com os MESMOS hiperparâmetros herdados e os MESMOS rótulos OOF em cache do
pipeline publicado (scripts/distill.py + scripts/extension_ctr23.py),
sementes canônicas {0,1,2}, e guarda o CRPS POR LINHA do hold-out:

  results/distillation/perline/<ds>.parquet
      y, twin (gêmeo exato de covariáveis no pool),
      crps_hard_s{0,1,2}, crps_soft_s{0,1,2}

Depois, bootstrap pareado das linhas (B=10.000, semente 0) da diferença
média de CRPS (controle − destilado) e do ganho relativo:

  results/distillation/bootstrap_lines.csv

Vereditos: vitória clara (IC95 > 0), derrota clara (IC95 < 0), empate
(IC95 contém 0). Nenhuma célula agregada publicada é alterada.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, OOF_DIR, REG_DATASETS, fit_xgb_quant_hard,
                     fit_xgb_quant_soft, load_xgb_params, ordinal_encode,
                     prep_pool, sort_quantiles)
from src.data.download import download_dataset
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

EXT_DIR = REPO / "results" / "distillation" / "extension"
PERLINE_DIR = REPO / "results" / "distillation" / "perline"
OUT_CSV = REPO / "results" / "distillation" / "bootstrap_lines.csv"
SEEDS = (0, 1, 2)
B = 10_000

# Mesma população da Tabela 7.x (paper_retention.csv); wine/cps88wages
# primeiro porque alimentam as correções de texto mais urgentes.
DATASETS = ["wine_quality", "cps88wages", "miami_housing", "space_ga",
            "health_insurance", "online_news", "california_housing",
            "kin8nm", "abalone", "cpu_activity", "grid_stability",
            "pumadyn32nh", "fps_benchmark", "fifa", "kings_county",
            "physiochemical_protein", "video_transcoding", "diamonds_real",
            "sarcos", "year_prediction"]

qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
TAUS = np.array(QUANTILES)


def perline_crps(y, Q):
    """CRPS por linha: 2x a média das pinball sobre a grade de 19 níveis."""
    diff = y[:, None] - Q                               # (n, 19)
    pin = np.maximum(TAUS * diff, (TAUS - 1.0) * diff)  # (n, 19)
    return 2.0 * pin.mean(axis=1)


def params_for(ds):
    if ds in REG_DATASETS:
        return load_xgb_params(ds)
    import json
    return json.loads((EXT_DIR / f"xgboost_{ds}.json").read_text())["best_params"]


def twin_mask(X_pool, X_test):
    pool_keys = set(map(tuple, X_pool.astype(str).values.tolist()))
    return np.array([tuple(r) in pool_keys
                     for r in X_test.astype(str).values.tolist()])


def build(ds, exp_cfg):
    out = PERLINE_DIR / f"{ds}.parquet"
    if out.exists():
        print(f"[skip] {ds}: perline já existe", flush=True)
        return
    if ds not in REG_DATASETS:
        download_dataset(ds)
    X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
    oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
    assert len(oof) == len(y_pool), f"{ds}: OOF desalinhado"
    assert np.allclose(oof["y"].to_numpy(), np.asarray(y_pool, float),
                       rtol=1e-9, atol=1e-9), f"{ds}: OOF y != pool y"
    Xp, Xt = ordinal_encode(X_pool, X_test)
    params = params_for(ds)
    cols = {"y": np.asarray(y_test, float),
            "twin": twin_mask(X_pool, X_test)}
    for seed in SEEDS:
        for arm, fit in (("hard", lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed)),
                         ("soft", lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed))):
            set_seed(seed)
            t0 = time.perf_counter()
            model = fit()
            Q = sort_quantiles(np.asarray(model.predict(Xt)))
            cols[f"crps_{arm}_s{seed}"] = perline_crps(cols["y"], Q)
            print(f"  [{ds} {arm} seed={seed}] fit={time.perf_counter()-t0:.0f}s "
                  f"crps={cols[f'crps_{arm}_s{seed}'].mean():.6f}", flush=True)
    PERLINE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(out)


def analyse():
    rows = []
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        p = PERLINE_DIR / f"{ds}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        hard = d[[f"crps_hard_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()
        soft = d[[f"crps_soft_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()
        diff = hard - soft                      # >0: destilado melhor
        n = len(diff)
        idx = rng.integers(0, n, size=(B, n))
        bs_diff = diff[idx].mean(axis=1)
        bs_gain = bs_diff / hard[idx].mean(axis=1)
        lo, hi = np.percentile(bs_diff, [2.5, 97.5])
        glo, ghi = np.percentile(100 * bs_gain, [2.5, 97.5])
        verdict = "vitoria" if lo > 0 else ("derrota" if hi < 0 else "empate")
        rows.append(dict(
            dataset=ds, n_test=n, twin_frac=round(float(d["twin"].mean()), 4),
            crps_hard=hard.mean(), crps_soft=soft.mean(),
            diff_mean=diff.mean(), ci_lo=lo, ci_hi=hi,
            gain_pct=100 * diff.mean() / hard.mean(),
            gain_ci_lo=glo, gain_ci_hi=ghi, verdict=verdict,
            # recorte por gêmeos (diagnóstico de duplicatas)
            gain_pct_twin=(100 * diff[d.twin].mean() / hard[d.twin].mean()
                           if d.twin.any() else np.nan),
            gain_pct_notwin=(100 * diff[~d.twin].mean() / hard[~d.twin].mean()
                             if (~d.twin).any() else np.nan),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    v = out.verdict.value_counts()
    print(f"\n=== bootstrap por linhas (B={B}) ===")
    print(out[["dataset", "twin_frac", "gain_pct", "gain_ci_lo", "gain_ci_hi",
               "verdict"]].round(3).to_string(index=False))
    print(f"\nplacar: {v.to_dict()}  → {OUT_CSV}")


if __name__ == "__main__":
    exp_cfg = load_experiment_config()
    for ds in DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        build(ds, exp_cfg)
    analyse()
    print("PERLINE DONE", flush=True)
