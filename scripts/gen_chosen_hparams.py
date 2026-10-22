#!/usr/bin/env python
"""Gera as tabelas de hiperparâmetros ESCOLHIDOS por conjunto (Apêndice B).

Complementa os espaços de busca (Tabelas B.1--B.4): para cada um dos 12
modelos ajustados, uma tabela com o valor vencedor do Optuna em cada um
dos 18 conjuntos, lido de ``results/raw/<modelo>_<dataset>.json``
(campo ``best_params`` — o mesmo artefato que o refit final consome).
Fecha a pendência registrada na reunião de 15/10/2026 ("os valores
escolhidos por conjunto não estão impressos").

Saída: ``dissertacao/appendix/hiperparametros-escolhidos.tex``
(incluído por ``apendice-b-espacos-busca.tex`` via ``\\input``).

Uso:  uv run python scripts/gen_chosen_hparams.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "dissertacao" / "appendix" / "hiperparametros-escolhidos.tex"

# Ordem canônica (idem gen_appendix_tables), só os ajustados.
MODELS = [
    ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"),
    ("mlp", "MLP"), ("realmlp", "RealMLP"), ("ft_transformer", "FT-Transformer"),
    ("saint", "SAINT"), ("stab", "STab"), ("tabnet", "TabNet"),
    ("tabm", "TabM"), ("kan", "KAN"), ("tabkan", "TabKAN"),
]


def fmt_val(v) -> str:
    """Valor de hiperparâmetro em notação compacta PT."""
    if isinstance(v, bool):
        return r"\texttt{" + str(v).lower() + "}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 0.01:
            s = f"{v:.3f}".rstrip("0").rstrip(".")
            return s.replace(".", "{,}")
        # notação científica compacta: 1,9e-8
        s = f"{v:.1e}"
        mant, exp = s.split("e")
        return mant.replace(".", "{,}") + r"\,e" + str(int(exp))
    return r"\texttt{" + str(v).replace("_", r"\_") + "}"


def tex_param(name: str) -> str:
    return r"\rotatebox{90}{\texttt{" + name.replace("_", r"\_") + "}}"


def tex_dataset(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def build_model_table(model: str, disp: str) -> str:
    files = sorted(RAW.glob(f"{model}_*.json"))
    rows = {}
    param_order: list[str] = []
    for f in files:
        d = json.loads(f.read_text())
        rows[d["dataset"]] = d["best_params"]
        for k in d["best_params"]:
            if k not in param_order:
                param_order.append(k)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption[Hiperparâmetros escolhidos --- " + disp + r"]{"
             + disp + r": configuração vencedora do Optuna em cada "
             r"conjunto (consumida pelo refit final e pelos \emph{folds} "
             r"externos). Fonte: \texttt{results/raw/"
             + model.replace("_", r"\_")
             + r"\_<conjunto>.json}, campo \texttt{best\_params}.}",
             r"\label{tab:app-hp-" + model.replace("_", "-") + "}",
             r"\begin{tabular}{l" + "r" * len(param_order) + "}",
             r"\toprule",
             " & ".join([r"\emph{Conjunto}"]
                        + [tex_param(p) for p in param_order]) + r" \\",
             r"\midrule"]
    for ds in sorted(rows):
        cells = [tex_dataset(ds)] + [
            fmt_val(rows[ds][p]) if p in rows[ds] else "---"
            for p in param_order
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    blocks = [
        "% ============================================================\n"
        "% ARQUIVO GERADO — NÃO EDITAR À MÃO.\n"
        "% Gerado por scripts/gen_chosen_hparams.py a partir de\n"
        "% results/raw/<modelo>_<dataset>.json (campo best_params).\n"
        "% Regenerar: uv run python scripts/gen_chosen_hparams.py\n"
        "% ============================================================\n"
    ]
    for model, disp in MODELS:
        blocks.append(build_model_table(model, disp))

    sys.path.insert(0, str(ROOT / "scripts"))
    from fix_tt_breaks import process as _ttbreaks
    texto, _ = _ttbreaks("\n\n".join(blocks) + "\n")
    OUT.write_text(texto, encoding="utf-8")
    print(f"OK: {OUT.relative_to(ROOT)} escrito ({len(MODELS)} tabelas).")


if __name__ == "__main__":
    main()
