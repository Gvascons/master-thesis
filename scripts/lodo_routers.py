#!/usr/bin/env python3
"""E5 (pré-registrado em docs/desenho-melhorias-finais.md, 14/10/2026):
o resultado negativo do roteamento por meta-características depende da
classe de roteador?

Mesmo protocolo LODO de scripts/lodo_validation.py, com roteadores
adicionais: árvore d=2 (referência), árvore d=3, kNN (k=3,
meta-características padronizadas) e regressão logística multinomial
(C=1, padronizada). Grades @11 (núcleo) e @14 (completa).

H-R1: nenhum roteador supera o baseline de classe majoritária em
nenhuma das duas grades.

Saída: results/aggregated/lodo_routers.csv
Uso: uv run python scripts/lodo_routers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.registry import load_dataset
from src.evaluation.enrichments import build_meta_features, winning_family_table
from src.models.factory import get_model_family

FEATS = ["log_n", "log_feat", "imbalance", "cat_frac",
         "is_binary", "is_multiclass", "is_regression"]
GEN2026 = {"kan", "tabkan", "tabfm"}

ROUTERS = {
    "tree_d2": lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
    "tree_d3": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
    "knn3": lambda: make_pipeline(StandardScaler(),
                                  KNeighborsClassifier(n_neighbors=3)),
    "logreg": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=5000)),
}


def build_W(t):
    fam_map = {m: get_model_family(m) for m in t.model.unique()}
    meta = build_meta_features(sorted(t.dataset.unique()), load_dataset,
                               datasets_yaml=str(REPO / "configs" / "datasets.yaml"))
    W = winning_family_table(t, fam_map, meta).copy()
    W["imbalance"] = W["imbalance"].fillna(1.0)
    W["log_n"] = np.log10(W["n"])
    W["log_feat"] = np.log10(W["n_feat"])
    for tk in ("binary", "multiclass", "regression"):
        W[f"is_{tk}"] = (W["task"] == tk).astype(int)
    return W


def main():
    t_full = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    grids = {"14": t_full, "11": t_full[~t_full.model.isin(GEN2026)]}

    rows = []
    for grid, t in grids.items():
        W = build_W(t)
        baseline = W.win_family.value_counts(normalize=True).max()
        for router, mk in ROUTERS.items():
            hits = []
            for left_out in W.index:
                tr = W.drop(index=left_out)
                clf = mk()
                clf.fit(tr[FEATS].values, tr["win_family"].values)
                pred = clf.predict(W.loc[[left_out], FEATS].values)[0]
                hits.append(pred == W.loc[left_out, "win_family"])
            rows.append({"grid": grid, "router": router,
                         "hit_rate": round(float(np.mean(hits)), 4),
                         "baseline_majority": round(float(baseline), 4),
                         "beats_baseline": bool(np.mean(hits) > baseline),
                         "n_datasets": len(W)})
            print(f"[@{grid}] {router:8s} hit={np.mean(hits):.3f} "
                  f"(baseline {baseline:.3f}) "
                  f"{'SUPERA' if np.mean(hits) > baseline else 'nao supera'}",
                  flush=True)

    out = REPO / "results/aggregated/lodo_routers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nH-R1 {'REFUTADA' if any(r['beats_baseline'] for r in rows) else 'CONFIRMADA'}: "
          f"{sum(r['beats_baseline'] for r in rows)}/{len(rows)} células superam o baseline")
    print(f"salvo: {out}")


if __name__ == "__main__":
    main()
