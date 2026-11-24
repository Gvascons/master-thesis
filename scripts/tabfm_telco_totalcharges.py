"""TabFM no telco com TotalCharges NUMÉRICO (pergunta 18 da arguição).

Contexto (auditoria externa de 23/11): o OpenML entrega TotalCharges como
texto e o pipeline o trata como categórica de 6.531 níveis — na
codificação ordinal, a maioria do hold-out recebe código de categoria
desconhecida. Nos GBDTs o efeito medido é de milésimos de AUC; para o
TabFM (justamente o 14º de 14 nesse conjunto, o pior caso da política
"sempre-TabFM") não havia medição. Este script repete o ajuste final do
benchmark (mesmo caminho da Fase 3 do scripts/train.py, mesma semente)
em DOIS braços: pipeline publicado (TotalCharges categórica) e corrigido
(pd.to_numeric, NaN onde o texto não é número), para TabFM e, como
âncora de comparação, XGBoost.

Saída: results/aggregated/telco_totalcharges_sensitivity.csv
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import json

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.models.factory import create_model, get_model_family
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "aggregated" / "telco_totalcharges_sensitivity.csv"
DS = "telco_customer_churn"


def run(model_name, fix_totalcharges, exp, pub):
    X, y, info = load_dataset(DS)
    variant = "corrigido(numerico)" if fix_totalcharges else "publicado(categorico)"
    if fix_totalcharges:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        info.cat_columns = [c for c in info.cat_columns if c != "TotalCharges"]
        if "TotalCharges" not in info.num_columns:
            info.num_columns = list(info.num_columns) + ["TotalCharges"]
    seed = exp.seed
    Xp, yp, Xt, yt = get_holdout_split(X, y, info, seed=seed,
                                       test_size=exp.test_size)
    Xtr, Xva, ytr, yva = train_test_split(Xp, yp, test_size=0.1,
                                          random_state=seed, stratify=yp)
    prep = get_preprocessor(get_model_family(model_name))(
        Xtr, info, X_val=Xva, X_test=Xt)
    kwargs = {}
    if model_name == "xgboost":
        kwargs = dict(json.loads(
            (REPO / "results/raw" / f"xgboost_{DS}.json").read_text()
        )["best_params"])
    if prep.cat_feature_indices is not None:
        kwargs["cat_feature_indices"] = prep.cat_feature_indices
    set_seed(seed)
    m = create_model(model_name, info.task_type, info.n_classes, seed=seed,
                     **kwargs)
    m.fit(prep.X_train, ytr, prep.X_val, yva)
    p = m.predict_proba(prep.X_test)
    auc = roc_auc_score(yt, p[:, 1])
    row = dict(model=model_name, variant=variant, auc=auc,
               logloss=log_loss(yt, p),
               publicado_auc=float(pub.loc[(model_name), "roc_auc"]))
    print(row, flush=True)
    del m
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return row


def main():
    exp = load_experiment_config()
    pub = (pd.read_csv(REPO / "results/aggregated/test_results.csv")
           .query("dataset==@DS").set_index("model"))
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    donekeys = {(r["model"], r["variant"]) for r in rows}
    models = tuple(sys.argv[1:]) or ("xgboost", "tabfm")
    for model_name in models:
        for fix in (False, True):
            variant = ("corrigido(numerico)" if fix
                       else "publicado(categorico)")
            if (model_name, variant) in donekeys:
                continue
            rows.append(run(model_name, fix, exp, pub))
            pd.DataFrame(rows).to_csv(OUT, index=False)
    d = pd.DataFrame(rows)
    piv = d.pivot(index="model", columns="variant", values="auc")
    piv["delta"] = piv["corrigido(numerico)"] - piv["publicado(categorico)"]
    print("\n", piv.round(5).to_string())
    print("TELCO TOTALCHARGES DONE", flush=True)


if __name__ == "__main__":
    main()
