"""Sensibilidade ao checkpoint do classificador TabPFN v2.5.

Contexto (correção de 23/11/2026, auditoria externa): o wrapper do
benchmark instancia `TabPFNClassifier()` sem `model_path`; no pacote
`tabpfn` 6.4.1 o padrão é `tabpfn-v2.5-classifier-v2.5_default.ckpt`, que
o model card oficial descreve como "finetuned on real-data" (Real-TabPFN;
43 conjuntos — a lista do Apêndice C.1 do arXiv 2511.08667 INCLUI o
`higgs`, OpenML 23512, presente neste benchmark). O texto do cap. 4 dizia
"exclusivamente sintético" — errado para o classificador; o REGRESSOR
padrão é sintético puro, então a QP3 não é afetada.

Este script repete o ajuste final do benchmark (mesmo caminho do
scripts/train.py, Fase 3, mesma semente) com o checkpoint padrão e com o
`tabpfn-v2.5-classifier-v2.5_default-2.ckpt` (melhor checkpoint SÓ
sintético, segundo o model card; baixável do repositório HF
Prior-Labs/tabpfn_2_5 — cópia local em results/ckpt/, fora do git) e
compara a AUC do hold-out. Uso:

  uv run python scripts/tabpfn_checkpoint_sensitivity.py credit_g phoneme ...

Saída: results/aggregated/tabpfn_checkpoint_sensitivity.csv
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

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.models.factory import get_model_family
from src.models.tabpfn_model import TabPFNModel
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "aggregated" / "tabpfn_checkpoint_sensitivity.csv"
CKPT_SYN = str(REPO / "results" / "ckpt" / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt")
CKPTS = {"default(real-finetuned)": None, "default-2(sintetico)": CKPT_SYN}


class PinnedTabPFN(TabPFNModel):
    ckpt = None

    def _create_tabpfn(self):
        from tabpfn import TabPFNClassifier
        kw = dict(device="auto", n_estimators=self.n_estimators,
                  random_state=self.seed)
        if self.ckpt:
            kw["model_path"] = self.ckpt
        return TabPFNClassifier(**kw)


def main():
    exp = load_experiment_config()
    seed = exp.seed
    pub = (pd.read_csv(REPO / "results/aggregated/test_results.csv")
           .query("model=='tabpfn'").set_index("dataset"))
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["dataset"], r["ckpt"]) for r in rows}
    for ds in sys.argv[1:]:
        X, y, info = load_dataset(ds)
        ms = getattr(exp, "max_dataset_samples", None)
        if ms and len(X) > ms:
            X, _, y, _ = train_test_split(X, y, train_size=ms,
                                          random_state=seed, stratify=y)
        Xp, yp, Xt, yt = get_holdout_split(X, y, info, seed=seed,
                                           test_size=exp.test_size)
        Xtr, Xva, ytr, yva = train_test_split(Xp, yp, test_size=0.1,
                                              random_state=seed, stratify=yp)
        prep = get_preprocessor(get_model_family("tabpfn"))(
            Xtr, info, X_val=Xva, X_test=Xt)
        for name, ck in CKPTS.items():
            if (ds, name) in done:
                continue
            set_seed(seed)
            PinnedTabPFN.ckpt = ck
            kw = {}
            if prep.cat_feature_indices is not None:
                kw["cat_feature_indices"] = prep.cat_feature_indices
            m = PinnedTabPFN(info.task_type, info.n_classes, seed=seed, **kw)
            m.fit(prep.X_train, ytr, prep.X_val, yva)
            p = m.predict_proba(prep.X_test)
            auc = (roc_auc_score(yt, p[:, 1]) if p.shape[1] == 2
                   else roc_auc_score(yt, p, multi_class="ovr"))
            r = dict(dataset=ds, ckpt=name, auc=auc,
                     logloss=log_loss(yt, p),
                     publicado_auc=float(pub.loc[ds, "roc_auc"]))
            print(r, flush=True)
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            del m
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    print("CHECKPOINT SENSITIVITY DONE", flush=True)


if __name__ == "__main__":
    main()
