#!/usr/bin/env python
"""Gera as tabelas completas de resultados do Apêndice A da dissertação.

Lê ``results/aggregated/test_results.csv`` (métrica primária por célula
modelo x dataset no hold-out de teste) e escreve
``dissertacao/appendix/tabelas-geradas.tex``, incluído pelo
``appendix/apendice-a-tabelas.tex`` via ``\\input``.

Convenções (documentadas na prosa do apêndice):
- uma tabela por tarefa, com a métrica primária da tarefa:
  binária = AUC-ROC (maior é melhor); multiclasse = log-loss (menor é
  melhor); regressão = RMSE (menor é melhor);
- linhas = datasets (ordem alfabética), colunas = os 14 modelos na ordem
  canônica do benchmark;
- melhor valor da linha em negrito; células estruturalmente ausentes
  (helena x TabPFN/TabFM) como "---";
- 4 casas decimais, separador decimal PT via ``{,}``;
- layout: ``sidewaystable`` (pacote rotating, já no preâmbulo) com
  cabeçalhos de modelo rotacionados 90 graus — 15 colunas não cabem em
  retrato; ver decisão registrada no apêndice.

Uso:  uv run python scripts/gen_appendix_tables.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "aggregated" / "test_results.csv"
OUT = ROOT / "dissertacao" / "appendix" / "tabelas-geradas.tex"

# Ordem canônica e nomes de exibição dos 14 modelos.
MODELS = [
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("mlp", "MLP"),
    ("realmlp", "RealMLP"),
    ("ft_transformer", "FT-Transformer"),
    ("saint", "SAINT"),
    ("stab", "STab"),
    ("tabnet", "TabNet"),
    ("tabm", "TabM"),
    ("kan", "KAN"),
    ("tabkan", "TabKAN"),
    ("tabpfn", "TabPFN"),
    ("tabfm", "TabFM"),
]

# (task_type, coluna da métrica, maior é melhor?, nome da métrica, label)
TASKS = [
    ("binary", "roc_auc", True, "AUC-ROC", "binaria"),
    ("multiclass", "log_loss", False, "log-loss", "multiclasse"),
    ("regression", "rmse", False, "RMSE", "regressao"),
]

TASK_TITLES = {
    "binaria": "Classificação binária --- AUC-ROC no teste "
    "(maior é melhor)",
    "multiclasse": "Classificação multiclasse --- log-loss no teste "
    "(menor é melhor)",
    "regressao": "Regressão --- RMSE no teste (menor é melhor)",
}


def fmt(v: float) -> str:
    """4 casas decimais, vírgula decimal PT protegida por chaves."""
    return f"{v:.4f}".replace(".", "{,}")


def tex_dataset(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def build_table(df: pd.DataFrame, task: str, metric: str,
                higher: bool, metric_name: str, label: str) -> str:
    sub = df[df["task_type"] == task]
    pivot = sub.pivot(index="dataset", columns="model", values=metric)
    datasets = sorted(pivot.index)

    lines = []
    lines.append(r"\begin{sidewaystable}[p]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption[" + TASK_TITLES[label].split(" --- ")[0]
        + r" --- resultados completos]{"
        + TASK_TITLES[label]
        + r". Métrica primária por modelo em cada \emph{dataset}; "
        + r"\textbf{negrito} = melhor da linha; ``---'' = célula "
        + r"estruturalmente ausente. Fonte: "
        + r"\texttt{results/aggregated/test\_results.csv}.}"
    )
    lines.append(r"\label{tab:app-" + label + "}")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODELS) + "}")
    lines.append(r"\toprule")
    header = [r"\emph{Dataset}"] + [
        r"\rotatebox{90}{" + disp + "}" for _, disp in MODELS
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for ds in datasets:
        row = pivot.loc[ds]
        vals = {m: row.get(m) for m, _ in MODELS}
        present = {m: v for m, v in vals.items() if pd.notna(v)}
        best = (max if higher else min)(present.values())
        cells = [tex_dataset(ds)]
        for m, _ in MODELS:
            v = vals[m]
            if pd.isna(v):
                cells.append("---")
            elif v == best:
                cells.append(r"\textbf{" + fmt(v) + "}")
            else:
                cells.append(fmt(v))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV)
    expected = {m for m, _ in MODELS}
    found = set(df["model"].unique())
    assert found == expected, f"modelos inesperados: {found ^ expected}"

    blocks = [
        "% ============================================================\n"
        "% ARQUIVO GERADO — NÃO EDITAR À MÃO.\n"
        "% Gerado por scripts/gen_appendix_tables.py a partir de\n"
        "% results/aggregated/test_results.csv.\n"
        "% Regenerar: uv run python scripts/gen_appendix_tables.py\n"
        "% ============================================================\n"
    ]
    n_rows = {}
    for task, metric, higher, metric_name, label in TASKS:
        blocks.append(build_table(df, task, metric, higher,
                                  metric_name, label))
        n_rows[label] = df[df["task_type"] == task]["dataset"].nunique()

    OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    total = len(df)
    print(f"OK: {OUT.relative_to(ROOT)} escrito.")
    print(f"Células lidas: {total} (esperado 250 = 252 - 2 exclusões "
          "estruturais helena x TabPFN/TabFM).")
    for label, n in n_rows.items():
        print(f"  tabela {label}: {n} datasets (linhas).")


if __name__ == "__main__":
    main()
