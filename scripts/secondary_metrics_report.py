#!/usr/bin/env python3
"""Métricas de matriz de confusão no hold-out (XGBoost x TabPFN x TabFM).

Motivação (reunião de acompanhamento de 04/11/2026): além das métricas
threshold-free que ancoram o benchmark (ROC-AUC, log-loss), reportar as
métricas operacionais que dependem de limiar --- precisão, recall
(sensibilidade), especificidade, F1 e acurácia balanceada --- com a classe
MINORITÁRIA como classe de interesse, para tornar explícito o trade-off
falso positivo x falso negativo que a AUC resume mas não expõe.

Lê os parquets por instância do E7 (results/complementarity/preds_*.parquet,
mesmo hold-out 20% semente 42, células validadas pelo gate |d| <= 1e-3) e
produz results/aggregated/secondary_metrics_holdout.csv.

Decisão declarada: o limiar é o argmax das probabilidades (equivalente a
p >= 0,5 nas binárias) --- o default de operação sem calibração de limiar.
Nenhum limiar é otimizado; o objetivo é descrever, não ranquear.

Sanidade: a acurácia recomputada deve bater com a salva em
results/aggregated/test_results.csv (|d| <= 1e-3, mesmo criterio do E7).

Uso: uv run python scripts/secondary_metrics_report.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SRC = REPO / "results/complementarity"
MODELS = ["xgboost", "tabpfn", "tabfm"]


def _fmt(v: float, best: bool) -> str:
    s = f"{v:.3f}".replace(".", "{,}")
    return f"\\textbf{{{s}}}" if best else s


def _texname(ds: str) -> str:
    return "\\texttt{" + ds.replace("_", "\\_\\allowbreak ") + "}"


def write_latex(out: pd.DataFrame) -> None:
    """Gera a tabela do Apêndice A (nenhum valor transcrito à mão)."""
    models = [m for m in MODELS]
    lines = [
        "% ============================================================",
        "% ARQUIVO GERADO - NAO EDITAR A MAO.",
        "% Gerado por scripts/secondary_metrics_report.py a partir de",
        "% results/aggregated/secondary_metrics_holdout.csv.",
        "% Regenerar: uv run python scripts/secondary_metrics_report.py",
        "% ============================================================",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption[Métricas operacionais no \\emph{hold-out} --- binárias]"
        "{Métricas operacionais no \\emph{hold-out} (limiar 0,5; classe de"
        " interesse = minoritária). Fonte: \\texttt{secondary\\_metrics\\_holdout.csv}.}",
        "\\label{tab:app-operacionais}",
        "\\begin{tabular}{lr ccc ccc ccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Precisão (min.)} & \\multicolumn{3}{c}{Recall (min.)}"
        " & \\multicolumn{3}{c}{Acurácia balanceada} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "\\emph{Conjunto} & \\%min. & XGB & PFN & FM & XGB & PFN & FM & XGB & PFN & FM \\\\",
        "\\midrule",
    ]
    bi = out[out.task_type == "binary"]
    for ds in sorted(bi.dataset.unique()):
        sub = bi[bi.dataset == ds].set_index("model").reindex(models)
        frac = f"{100 * sub.minority_frac.iloc[0]:.1f}".replace(".", "{,}")
        cells = []
        for col in ("precision_min", "recall_min", "balanced_accuracy"):
            vals = sub[col].values
            best = np.nanmax(vals)
            cells += [_fmt(v, abs(v - best) < 5e-4) for v in vals]
        lines.append(f"{_texname(ds)} & {frac} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\midrule",
        " & & \\multicolumn{3}{c}{Precisão macro} & \\multicolumn{3}{c}{Recall macro}"
        " & \\multicolumn{3}{c}{F1 macro} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
    ]
    mc = out[out.task_type == "multiclass"]
    for ds in sorted(mc.dataset.unique()):
        sub = mc[mc.dataset == ds].set_index("model").reindex(models)
        if sub.accuracy.isna().sum() > 1:   # helena: só XGBoost
            continue
        cells = []
        for col in ("precision_macro", "recall_macro", "f1_macro"):
            vals = sub[col].values
            best = np.nanmax(vals)
            cells += [_fmt(v, abs(v - best) < 5e-4) for v in vals]
        lines.append(f"{_texname(ds)} & --- & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    dest = REPO / "dissertacao/appendix/tabela-metricas-operacionais.tex"
    dest.write_text("\n".join(lines))
    print(f"LaTeX -> {dest}")


def main() -> None:
    sanity = pd.read_csv(SRC / "sanity_check.csv")
    ok = {(r.model, r.dataset) for r in sanity.itertuples() if r.gate == "OK"}
    meta = {r.dataset: r.task_type for r in sanity.itertuples() if r.gate == "OK"}
    test = pd.read_csv(REPO / "results/aggregated/test_results.csv")

    rows, fails = [], []
    for ds, task in sorted(meta.items()):
        if task not in ("binary", "multiclass"):
            continue
        for model in MODELS:
            if (model, ds) not in ok:
                continue
            df = pd.read_parquet(SRC / f"preds_{model}_{ds}.parquet")
            y, yp = df.y_true.values, df.y_pred.values
            acc = float((y == yp).mean())

            ref = test[(test.model == model) & (test.dataset == ds)]
            acc_ref = float(ref.accuracy.iloc[0]) if len(ref) else np.nan
            delta = abs(acc - acc_ref) if np.isfinite(acc_ref) else np.nan
            gate = "OK" if (np.isfinite(delta) and delta <= 1e-3) else f"DELTA={delta:.2e}"
            if gate != "OK":
                fails.append((model, ds, gate))

            row = {"model": model, "dataset": ds, "task_type": task,
                   "n_test": len(y), "accuracy": acc,
                   "accuracy_benchmark": acc_ref, "sanity_gate": gate,
                   "balanced_accuracy": float(balanced_accuracy_score(y, yp))}

            if task == "binary":
                counts = pd.Series(y).value_counts()
                minority = int(counts.idxmin())
                majority = int(counts.idxmax())
                row.update(
                    minority_class=minority,
                    minority_frac=float(counts.min() / counts.sum()),
                    precision_min=float(precision_score(y, yp, pos_label=minority,
                                                        zero_division=0)),
                    recall_min=float(recall_score(y, yp, pos_label=minority,
                                                  zero_division=0)),
                    f1_min=float(f1_score(y, yp, pos_label=minority,
                                          zero_division=0)),
                    specificity=float(recall_score(y, yp, pos_label=majority,
                                                   zero_division=0)),
                )
            else:
                row.update(
                    f1_macro=float(f1_score(y, yp, average="macro",
                                            zero_division=0)),
                    recall_macro=float(recall_score(y, yp, average="macro",
                                                    zero_division=0)),
                    precision_macro=float(precision_score(y, yp, average="macro",
                                                          zero_division=0)),
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    dest = REPO / "results/aggregated/secondary_metrics_holdout.csv"
    out.to_csv(dest, index=False)
    print(f"{len(out)} linhas -> {dest}")
    write_latex(out)
    if fails:
        print("ATENCAO - acurácia recomputada divergiu do benchmark:")
        for f in fails:
            print("  ", f)
    else:
        print("Sanidade: acurácia recomputada bate com test_results.csv em todas as células.")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
