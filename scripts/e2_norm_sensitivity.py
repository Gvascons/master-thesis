#!/usr/bin/env python3
"""E2 — análise de sensibilidade da normalização (adendo de 24/11/2026).

O E2 pré-registrado (scripts/lodo_weighted_regret.py) usa normalização
por POSTOS de acurácia e latência. O texto do capítulo 6 afirmava que
normalizações alternativas não mudam a leitura, sem artefato; este
script mede essa afirmação. Três normalizações, mesmas políticas e o
mesmo regret normalizado do estudo original:

- rank:   postos normalizados [0,1] (controle; reproduz o CSV original)
- log:    acurácia em minmax [0,1] por conjunto (0 = melhor); latência
          em minmax de log10(us/linha)
- linear: acurácia em minmax; latência em minmax dos us/linha brutos

Grade fina de w (passo 0,01). Saída:
results/aggregated/lodo_weighted_regret_norms.csv e, no stdout, as
medianas/médias por política x w (grade 0,05) e o ÚLTIMO w em que
sempre-FM ainda é a política regret-ótima, nas grades 0,05 e 0,01,
por normalização e estatística.

Uso: uv run python scripts/e2_norm_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRIMARY = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
           "regression": ("rmse", False)}
FAMILIES = {
    "xgboost": "GBDT", "lightgbm": "GBDT", "catboost": "GBDT",
    "ft_transformer": "DL", "tabnet": "DL", "saint": "DL", "stab": "DL",
    "tabm": "DL", "mlp": "DL", "realmlp": "DL", "kan": "DL", "tabkan": "DL",
    "tabpfn": "FM", "tabfm": "FM",
}
POLICIES = {"Always-GBDT": "GBDT", "Always-DL": "DL", "Always-FM": "FM"}
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    rng = (hi - lo) or 1.0
    return (s - lo) / rng


def norm_acc(s: pd.Series, higher: bool, how: str) -> pd.Series:
    if how == "rank":
        r = s.rank(ascending=not higher)
        return (r - 1) / (len(r) - 1)
    v = minmax(s)
    return 1.0 - v if higher else v  # 0 = melhor


def norm_lat(us: pd.Series, how: str) -> pd.Series:
    if how == "rank":
        r = us.rank()
        return (r - 1) / (len(r) - 1)
    if how == "log":
        return minmax(np.log10(us))
    return minmax(us)  # linear


def main():
    tr = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    lat = (pd.read_csv(REPO / "results/latency/latency_adult.csv")
           .set_index("model")["us_per_row"])

    rows = []
    for how in ("rank", "log", "linear"):
        lat_norm = norm_lat(lat, how)
        for ds, sub in tr.groupby("dataset"):
            task = sub.task_type.iloc[0]
            metric, higher = PRIMARY[task]
            s = sub.dropna(subset=[metric]).set_index("model")[metric]
            if len(s) < 4:
                continue
            acc_norm = norm_acc(s, higher, how)
            for w in WEIGHTS:
                score = (w * acc_norm
                         + (1 - w) * lat_norm.reindex(acc_norm.index)).dropna()
                fam = pd.Series({m: FAMILIES[m] for m in score.index})
                best, worst = score.min(), score.max()
                denom = (worst - best) or 1.0
                for pol, family in POLICIES.items():
                    in_fam = score[fam == family]
                    if in_fam.empty:
                        continue
                    regret = (in_fam.min() - best) / denom
                    rows.append({"norm": how, "dataset": ds, "task": task,
                                 "w_acc": w, "policy": pol,
                                 "regret": round(float(regret), 6)})
    df = pd.DataFrame(rows)
    # versionados: (a) resumo agregado na grade 0,01 (pequeno) e
    # (b) por conjunto na grade 0,05; a grade 0,01 por conjunto é
    # reproduzível por este script (determinístico sobre os artefatos).
    agg = (df.groupby(["norm", "policy", "w_acc"])["regret"]
           .agg(["median", "mean"]).reset_index())
    out_sum = REPO / "results/aggregated/lodo_weighted_regret_norms_summary.csv"
    agg.round(6).to_csv(out_sum, index=False)
    out = REPO / "results/aggregated/lodo_weighted_regret_norms.csv"
    df[df.w_acc.map(lambda w: round(w * 100) % 5 == 0)].to_csv(out, index=False)

    def last_fm_optimal(piv, grid):
        ws = [w for w in sorted(piv.index) if round(w * 100) % round(grid * 100) == 0]
        last = None
        for w in ws:  # crescente: guarda o menor w a partir do qual FM e otima ate 1
            if piv.loc[w, "Always-FM"] <= piv.loc[w].min() + 1e-9:
                if last is None or w < last:
                    # exige otimalidade continua ate w=1
                    tail = [x for x in ws if x >= w]
                    if all(piv.loc[x, "Always-FM"] <= piv.loc[x].min() + 1e-9
                           for x in tail):
                        last = w
        return last

    print("\n=== último w em que sempre-FM ainda é a política ótima ===")
    print("norm    | estat.  | grade 0,05 | grade 0,01 | peso na latência na inversão")
    for how in ("rank", "log", "linear"):
        a = agg[agg.norm == how]
        for stat in ("median", "mean"):
            piv = a.pivot(index="w_acc", columns="policy", values=stat)
            w05 = last_fm_optimal(piv, 0.05)
            w01 = last_fm_optimal(piv, 0.01)
            band = (f"{round((1 - w05) * 100)}--{round((1 - w05 + 0.05) * 100)}%"
                    if w05 is not None else "--")
            print(f"{how:7s} | {stat:7s} | {w05} | {w01} | {band}")
        med = a.pivot(index="w_acc", columns="policy", values="median")
        sub = med.loc[[w for w in med.index if round(w * 100) % 5 == 0]]
        print(f"--- {how}: mediana por w (grade 0,05) ---")
        print(sub.round(3).to_string())
    print(f"\nsalvos: {out.relative_to(REPO)} (por conjunto, grade 0,05)"
          f" + {out_sum.relative_to(REPO)} (resumo, grade 0,01)")


if __name__ == "__main__":
    main()
