#!/usr/bin/env python3
"""ERRATA 19/10/2026 — o dataset rotulado "superconduct" sempre foi o
OnlineNewsPopularity (OpenML 44064, suite "regression on categorical and
numerical features" do tabular-benchmark; alvo = log do numero de
compartilhamentos). Mesma classe de erro do episodio diamonds/kin8nm
(docs/errata-diamonds-kin8nm.md): o openml_id pinado na criacao do
projeto nunca correspondeu a identidade pretendida. A checagem
nome<->ID<->shape registrada como pendencia na licao 2 daquela errata
foi executada em 19/10/2026 sobre os 33 IDs do config e detectou este
caso (e o do california_housing binarizado, tratado por sensibilidade,
nao por rename — o rotulo dele esta correto).

Renomeia o rotulo em todos os artefatos (validade experimental intacta:
apenas o NOME estava errado). Idempotente: re-execucao nao altera nada.

Uso: uv run python scripts/errata_rename_online_news.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OLD, NEW = "superconduct", "online_news"
changed = []


def rename_file(p: Path) -> Path:
    q = p.with_name(p.name.replace(OLD, NEW))
    if p.exists() and p != q:
        p.rename(q)
        changed.append(f"mv {p.relative_to(REPO)} -> {q.name}")
        return q
    return q if q.exists() else p


def fix_csv(p: Path):
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "dataset" in df.columns and (df.dataset == OLD).any():
        df.loc[df.dataset == OLD, "dataset"] = NEW
        df.to_csv(p, index=False)
        changed.append(f"csv {p.relative_to(REPO)}")


def main():
    # 1) parquet bruto
    rename_file(REPO / "data/raw" / f"{OLD}.parquet")

    # 2) resultados brutos do benchmark (JSONs nomeados e campo interno)
    for p in sorted((REPO / "results/raw").glob(f"*{OLD}*")):
        q = rename_file(p)
        if q.suffix == ".json":
            d = json.loads(q.read_text())
            if d.get("dataset") == OLD:
                d["dataset"] = NEW
                q.write_text(json.dumps(d, indent=2))
                changed.append(f"json field {q.name}")

    # 3) colunas dataset dos CSVs agregados e de destilacao
    for sub in ("results/aggregated", "results/distillation"):
        for p in sorted((REPO / sub).glob("*.csv")):
            fix_csv(p)
    # fold_results.csv (local-only, ~300MB) via stream
    fr = REPO / "results/aggregated/fold_results.csv"
    if fr.exists():
        txt = fr.read_text()
        if OLD in txt:
            fr.write_text(txt.replace(OLD, NEW))
            changed.append("csv fold_results.csv (local)")

    # 4) parquets OOF da destilacao
    for p in sorted((REPO / "results/distillation/oof").glob(f"*{OLD}*")):
        rename_file(p)

    # 5) lodo_validation usa coluna dataset; demais matrizes modelo x modelo
    #    nao contem nomes de dataset (nada a fazer).

    print(f"{len(changed)} alteracoes:")
    for c in changed:
        print(" ", c)
    if not changed:
        print("  (nenhuma — ja renomeado)")


if __name__ == "__main__":
    main()
