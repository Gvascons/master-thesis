"""Redes em regressão com alvo padronizado (achado A4 da auditoria de 23/11).

O pipeline publicado treina as redes de regressão no alvo bruto; a
auditoria externa mediu (1 semente) que, com padronização do alvo, o MLP
sobe posições no year_prediction. Este script quantifica o efeito nas
cinco redes principais × cinco conjuntos de regressão do núcleo, com os
MESMOS hiperparâmetros do benchmark (sem re-busca — limitação declarada:
os HPs foram buscados para o alvo bruto, então a comparação é um PISO do
efeito da padronização) e o mesmo caminho da Fase 3 do scripts/train.py.
Alvo: z-score ajustado no treino final (90%), aplicado a treino/val;
predições des-padronizadas antes do RMSE. Sementes 42 (comparável ao
publicado), 43 e 44 (dispersão).

Saída: results/aggregated/dl_target_standardization.csv
Resumível: pula células já presentes no CSV.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.models.factory import create_model, get_model_family
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "aggregated" / "dl_target_standardization.csv"
MODELS = ["mlp", "realmlp", "ft_transformer", "saint", "tabm"]
DATASETS = ["wine_quality", "california_housing", "online_news", "kin8nm",
            "year_prediction"]
SEEDS = [42, 43, 44]


def main():
    exp = load_experiment_config()
    pub = (pd.read_csv(REPO / "results/aggregated/test_results.csv")
           .set_index(["model", "dataset"]))
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["model"], r["dataset"], r["seed"]) for r in rows}

    for ds in DATASETS:
        X, y, info = load_dataset(ds)
        ms = getattr(exp, "max_dataset_samples", None)
        if ms and len(X) > ms:
            X, _, y, _ = train_test_split(X, y, train_size=ms,
                                          random_state=exp.seed)
        Xp, yp, Xt, yt = get_holdout_split(X, y, info, seed=exp.seed,
                                           test_size=exp.test_size)
        for model_name in MODELS:
            jp = REPO / "results/raw" / f"{model_name}_{ds}.json"
            if not jp.exists():
                print(f"[skip] sem params para {model_name}/{ds}", flush=True)
                continue
            best = json.loads(jp.read_text()).get("best_params", {}) or {}
            best.pop("cat_feature_indices", None)
            for seed in SEEDS:
                if (model_name, ds, seed) in done:
                    continue
                # split final 90/10 com a MESMA semente 42 do benchmark
                # (a semente varrida controla o modelo, não a partição)
                Xtr, Xva, ytr, yva = train_test_split(
                    Xp, yp, test_size=0.1, random_state=exp.seed)
                prep = get_preprocessor(get_model_family(model_name))(
                    Xtr, info, X_val=Xva, X_test=Xt)
                mu, sd = float(np.mean(ytr)), float(np.std(ytr))
                sd = sd if sd > 0 else 1.0
                kw = dict(best)
                if prep.cat_feature_indices is not None:
                    kw["cat_feature_indices"] = prep.cat_feature_indices
                set_seed(seed)
                t0 = time.perf_counter()
                m = create_model(model_name, info.task_type, info.n_classes,
                                 seed=seed, **kw)
                m.fit(prep.X_train, (ytr - mu) / sd,
                      prep.X_val, (yva - mu) / sd)
                pred = np.asarray(m.predict(prep.X_test)) * sd + mu
                rmse = float(np.sqrt(np.mean((yt - pred) ** 2)))
                row = dict(model=model_name, dataset=ds, seed=seed,
                           rmse_scaled_target=rmse,
                           rmse_publicado_bruto=float(
                               pub.loc[(model_name, ds), "rmse"]),
                           fit_s=round(time.perf_counter() - t0, 1))
                print(row, flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT, index=False)
                del m
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    d = pd.DataFrame(rows)
    g = d.groupby(["dataset", "model"]).agg(
        rmse_scaled=("rmse_scaled_target", "mean"),
        rmse_bruto=("rmse_publicado_bruto", "first")).reset_index()
    g["melhora_%"] = 100 * (g.rmse_bruto - g.rmse_scaled) / g.rmse_bruto
    print("\n=== resumo (media das 3 sementes vs publicado) ===")
    print(g.round(4).to_string(index=False))
    print("DL TARGET STANDARDIZATION DONE", flush=True)


if __name__ == "__main__":
    main()
