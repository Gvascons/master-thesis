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

import sys
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
    lines.append(r"\setlength{\tabcolsep}{3pt}")
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
    header = [r"\emph{Conjunto}"] + [
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


# Métricas secundárias por tarefa: (coluna, nome exibido, maior é melhor?)
SECONDARY = {
    "binary": [("roc_auc", "AUC-ROC", True), ("accuracy", "Acurácia", True),
               ("f1", "F1", True), ("ks", "KS", True),
               ("log_loss", "log-loss", False)],
    "multiclass": [("log_loss", "log-loss", False),
                   ("accuracy", "Acurácia", True),
                   ("f1_macro", "F1-macro", True),
                   ("f1_weighted", "F1-ponderado", True),
                   ("roc_auc_ovr", "AUC-OvR", True)],
    "regression": [("rmse", "RMSE", False), ("mae", "MAE", False),
                   ("r2", "$R^{2}$", True)],
}
TASK_PT = {"binary": "binária", "multiclass": "multiclasse",
           "regression": "regressão"}


def fmt2(v: float) -> str:
    return f"{v:.2f}".replace(".", "{,}")


def build_secondary_rank_table(df: pd.DataFrame, task: str) -> str:
    """Rank médio (1 = melhor) por modelo em cada métrica da tarefa.

    Ranks tornam as métricas comensuráveis entre datasets (RMSE/MAE e
    log-loss têm escalas incomensuráveis entre conjuntos — a mesma razão
    pela qual o cap. 5 raciocina em ranks)."""
    sub = df[df["task_type"] == task]
    specs = SECONDARY[task]
    cols = {}
    for metric, disp, higher in specs:
        piv = sub.pivot(index="dataset", columns="model", values=metric)
        ranks = piv.rank(axis=1, ascending=not higher).mean()
        cols[disp] = ranks
    tab = pd.DataFrame(cols)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption[Métricas secundárias --- tarefa " + TASK_PT[task]
             + r"]{Rank médio (1 = melhor) dos 14 modelos em cada métrica "
             + r"da tarefa " + TASK_PT[task] + r"; a primária aparece como "
             + r"referência. Valores por célula em "
             + r"\texttt{results/aggregated/test\_results.csv}. Fonte: idem.}",
             r"\label{tab:app-secundarias-" + task + "}",
             r"\begin{tabular}{l" + "c" * len(specs) + "}", r"\toprule",
             " & ".join([r"\emph{Modelo}"] + [d for _, d, _ in specs])
             + r" \\", r"\midrule"]
    for m, disp in MODELS:
        cells = [disp]
        for _, dcol, _ in specs:
            v = tab.loc[m, dcol] if m in tab.index else float("nan")
            best = tab[dcol].min()
            s = fmt2(v)
            cells.append(r"\textbf{" + s + "}" if abs(v - best) < 1e-9 else s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_cohensd_table(which: str) -> str:
    """Matriz de d de Cohen pareado (linha - coluna) sobre a primária."""
    src = ROOT / "results" / "aggregated" / f"cohens_d_{which}.csv"
    mat = pd.read_csv(src, index_col=0)
    order = [m for m, _ in MODELS if m in mat.index]
    disp = {m: d for m, d in MODELS}
    mat = mat.loc[order, order]
    nome = "binária (AUC-ROC)" if which == "binary" else "regressão (RMSE)"
    direcao = (r"positivo = linha melhor (AUC maior)" if which == "binary"
               else r"positivo = linha \emph{pior} (RMSE maior)")
    lines = [r"\begin{sidewaystable}[p]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption[Tamanhos de efeito ($d$ de Cohen) --- "
             + ("binária" if which == "binary" else "regressão") + r"]{$d$ "
             + r"de Cohen pareado (linha $-$ coluna) na tarefa " + nome
             + r"; " + direcao + r". Fonte: \texttt{results/aggregated/"
             + f"cohens\\_d\\_{which}.csv" + r"}.}",
             r"\label{tab:app-cohensd-" + which + "}",
             r"\begin{tabular}{l" + "c" * len(order) + "}", r"\toprule",
             " & ".join([""] + [r"\rotatebox{90}{" + disp[m] + "}"
                                for m in order]) + r" \\", r"\midrule"]
    for m in order:
        cells = [disp[m]]
        for c in order:
            v = mat.loc[m, c]
            # modo matemático para o sinal de menos tipográfico correto
            cells.append("---" if m == c else "$" + fmt2(v) + "$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    return "\n".join(lines)


def build_ci_table() -> str:
    """IC95 bootstrap (B=10.000) da média de AUC-ROC na binária."""
    src = ROOT / "results" / "aggregated" / "bootstrap_ci.csv"
    ci = pd.read_csv(src)
    ci = ci[(ci.task_type == "binary") & (ci.metric == "roc_auc")]
    ci = ci.set_index("model")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption[IC bootstrap da média de AUC --- binária]{Média de "
             r"AUC-ROC na tarefa binária com IC95\% de \emph{bootstrap} "
             r"percentil ($B{=}10\,000$) sobre os conjuntos. Fonte: "
             r"\texttt{results/aggregated/bootstrap\_ci.csv}.}",
             r"\label{tab:app-bootstrap-auc}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"\emph{Modelo} & Média & IC95\% \\", r"\midrule"]
    best = ci["mean"].max()
    for m, disp in MODELS:
        if m not in ci.index:
            continue
        r = ci.loc[m]
        mean = fmt(r["mean"])
        if abs(r["mean"] - best) < 1e-12:
            mean = r"\textbf{" + mean + "}"
        lines.append(f"{disp} & {mean} & "
                     f"[{fmt(r['ci_lower'])}; {fmt(r['ci_upper'])}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_cost_table(df: pd.DataFrame) -> str:
    """Custos consolidados: treino, tuning (excl. adult) e latência."""
    lat = pd.read_csv(ROOT / "results" / "latency" / "latency_adult.csv")
    lat = lat.set_index("model")["us_per_row"]
    no_adult = df[df.dataset != "adult"]
    train = df.groupby("model")["train_time_s"].median()
    tune = no_adult.groupby("model")["tuning_time_s"].median()
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption[Custos consolidados por modelo]{Custos medianos por "
             r"modelo: treino (s, 18 conjuntos), tuning (s, excl.\ "
             r"\texttt{adult} --- Seção~\ref{sec:met-custo}) e latência "
             r"(\textmu s/linha, \texttt{adult}). Fonte: \texttt{test\_results.csv} "
             r"+ \texttt{latency\_adult.csv}.}",
             r"\label{tab:app-custos}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"\emph{Modelo} & Treino (s) & Tuning (s) & Latência "
             r"(\textmu s/linha) \\", r"\midrule"]
    for m, disp in MODELS:
        tr = f"{train[m]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        tu = ("---" if pd.isna(tune.get(m)) or tune.get(m, 0) == 0
              else f"{tune[m]:,.0f}".replace(",", "."))
        lt = f"{lat[m]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        lines.append(f"{disp} & {tr} & {tu} & {lt}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_foldvar_table() -> str:
    """Variabilidade entre folds: mediana do coeficiente de variação (%)."""
    src = ROOT / "results" / "aggregated" / "cv_summary.csv"
    cv = pd.read_csv(src, header=[0, 1], index_col=0)
    # pandas desduplica a 1a linha do cabecalho ("roc_auc.1" para o std)
    cv.columns = [f"{a.split('.')[0]}_{b}" if b in ("mean", "std") else a
                  for a, b in cv.columns]
    tasks = {"binary": "roc_auc", "multiclass": "log_loss",
             "regression": "rmse"}
    task_of = pd.read_csv(CSV)[["dataset", "task_type"]].drop_duplicates()
    task_of = task_of.set_index("dataset")["task_type"]
    out = {}
    for task, metric in tasks.items():
        mcol, scol = f"{metric}_mean", f"{metric}_std"
        sub = cv[cv["dataset"].map(task_of).eq(task)].copy()
        sub = sub.dropna(subset=[mcol, scol])
        sub["cvpct"] = 100 * sub[scol] / sub[mcol].abs()
        out[task] = sub.groupby("model")["cvpct"].median()
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption[Variabilidade entre \emph{folds}]{Mediana, sobre os "
             r"conjuntos da tarefa, do coeficiente de variação entre "
             r"\emph{folds} da métrica primária (\%; menor = mais estável). "
             r"Fonte: \texttt{results/aggregated/cv\_summary.csv}.}",
             r"\label{tab:app-foldvar}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"\emph{Modelo} & Binária (AUC) & Multiclasse (log-loss) & "
             r"Regressão (RMSE) \\", r"\midrule"]
    for m, disp in MODELS:
        cells = [disp]
        for task in tasks:
            v = out[task].get(m)
            cells.append("---" if v is None or pd.isna(v) else fmt2(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
        "% results/aggregated/{test_results,cohens_d_*,bootstrap_ci,\n"
        "% cv_summary}.csv e results/latency/latency_adult.csv.\n"
        "% Regenerar: uv run python scripts/gen_appendix_tables.py\n"
        "% ============================================================\n"
    ]
    n_rows = {}
    for task, metric, higher, metric_name, label in TASKS:
        blocks.append(build_table(df, task, metric, higher,
                                  metric_name, label))
        n_rows[label] = df[df["task_type"] == task]["dataset"].nunique()

    for task in ("binary", "multiclass", "regression"):
        blocks.append(build_secondary_rank_table(df, task))
    blocks.append(build_cohensd_table("binary"))
    blocks.append(build_cohensd_table("regression"))
    blocks.append(build_ci_table())
    blocks.append(build_cost_table(df))
    blocks.append(build_foldvar_table())

    # caminhos quebráveis nas legendas (auditoria de layout 19/10/2026)
    sys.path.insert(0, str(ROOT / "scripts"))
    from fix_tt_breaks import process as _ttbreaks
    texto, _ = _ttbreaks("\n\n".join(blocks) + "\n")
    OUT.write_text(texto, encoding="utf-8")
    total = len(df)
    print(f"OK: {OUT.relative_to(ROOT)} escrito ({len(blocks) - 1} tabelas).")
    print(f"Células lidas: {total} (esperado 250 = 252 - 2 exclusões "
          "estruturais helena x TabPFN/TabFM).")
    for label, n in n_rows.items():
        print(f"  tabela {label}: {n} datasets (linhas).")


if __name__ == "__main__":
    main()
