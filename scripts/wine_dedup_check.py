"""Teste causal das duplicatas do wine_quality (auditoria externa de 23/11).

27,6% do hold-out do wine_quality tem gêmeo exato (covariáveis + alvo) no
pool — o controle quantílico memoriza esses gêmeos e o aluno destilado,
treinado em curvas suavizadas do professor, não. Este script refaz o
experimento no conjunto DEDUPLICADO (drop_duplicates em linha inteira):
professor TabPFN re-rotulado OOF (K=5, semente 42), controle e destilado
com os hiperparâmetros herdados do benchmark, sementes {0,1,2}.

Saída: results/distillation/wine_dedup.csv (mesmo cabeçalho de células do
extension.csv) + retenção impressa. Nada do pipeline publicado é alterado.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, crps_from_quantiles, fit_teacher,
                     fit_xgb_quant_hard, fit_xgb_quant_soft, free_teacher,
                     interval_metrics, load_xgb_params, make_teacher,
                     ordinal_encode, sort_quantiles, teacher_predict)
from src.data.registry import get_holdout_split, load_dataset
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "distillation" / "wine_dedup.csv"
qcols = [f"q{int(t*100):02d}" for t in QUANTILES]


def main():
    X, y, info = load_dataset("wine_quality")
    frame = X.copy()
    frame["__y"] = np.asarray(y)
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    yd = frame.pop("__y").to_numpy(dtype=np.float64)
    Xd = frame
    print(f"wine_quality: {before} -> {len(Xd)} após drop_duplicates", flush=True)

    X_pool, y_pool, X_test, y_test = get_holdout_split(Xd, yd, info, seed=42,
                                                       test_size=0.2)
    Xp, Xt = ordinal_encode(X_pool, X_test)
    params = load_xgb_params("wine_quality")
    rows = []

    oof_path = REPO / "results" / "distillation" / "oof" / "tabpfn_wine_dedup.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        assert len(oof) == len(y_pool)
    else:
        n = len(y_pool)
        mean = np.full(n, np.nan)
        Q = np.full((n, len(QUANTILES)), np.nan)
        for tr, ho in KFold(n_splits=5, shuffle=True, random_state=42).split(Xp):
            m_ = make_teacher("auto")
            fit_teacher(m_, Xp[tr], y_pool[tr])
            m, q = teacher_predict(m_, Xp[ho])
            mean[ho], Q[ho] = m, q
            free_teacher(m_)
        assert not np.isnan(mean).any()
        oof = pd.DataFrame(Q, columns=qcols)
        oof.insert(0, "mean", mean)
        oof.insert(0, "y", y_pool)
        oof.to_parquet(oof_path)
    print("OOF ok", flush=True)

    m_ = make_teacher("auto")
    t0 = time.perf_counter()
    fit_teacher(m_, Xp, y_pool)
    tm, tQ = teacher_predict(m_, Xt)
    free_teacher(m_)
    rows.append(dict(dataset="wine_dedup", cell="teacher", seed=0,
                     rmse=float(np.sqrt(np.mean((y_test - tm) ** 2))),
                     crps=crps_from_quantiles(y_test, tQ),
                     picp80=interval_metrics(y_test, tQ)["picp80"],
                     time_s=round(time.perf_counter() - t0, 1)))
    print(f"teacher crps={rows[-1]['crps']:.6f}", flush=True)

    for seed in (0, 1, 2):
        for cell, fit in (("student_quant_hard",
                           lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed)),
                          ("student_quant_soft",
                           lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed))):
            set_seed(seed)
            t0 = time.perf_counter()
            model = fit()
            Q = sort_quantiles(np.asarray(model.predict(Xt)))
            rows.append(dict(dataset="wine_dedup", cell=cell, seed=seed,
                             rmse=float(np.sqrt(np.mean(
                                 (y_test - Q[:, QUANTILES.index(0.50)]) ** 2))),
                             crps=crps_from_quantiles(y_test, Q),
                             picp80=interval_metrics(y_test, Q)["picp80"],
                             time_s=round(time.perf_counter() - t0, 1)))
            print(f"[{cell} seed={seed}] crps={rows[-1]['crps']:.6f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    g = df.groupby("cell").crps.mean()
    ret = (g.student_quant_hard - g.student_quant_soft) / (g.student_quant_hard - g.teacher)
    gain = 100 * (g.student_quant_hard - g.student_quant_soft) / g.student_quant_hard
    per_seed = df[df.cell != "teacher"].pivot(index="seed", columns="cell", values="crps")
    wins = (per_seed.student_quant_soft < per_seed.student_quant_hard).sum()
    print(f"\nDEDUP: teacher={g.teacher:.6f} hard={g.student_quant_hard:.6f} "
          f"soft={g.student_quant_soft:.6f} | retencao={ret:.4f} "
          f"ganho={gain:.2f}% | destilado vence em {wins}/3 sementes")
    print(f"→ {OUT}\nWINE DEDUP DONE", flush=True)


if __name__ == "__main__":
    main()
