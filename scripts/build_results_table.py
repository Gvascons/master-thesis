#!/usr/bin/env python3
"""Build the master results table (a presentable, all-in-one Markdown report).

Reads the aggregated hold-out results, the latency measurements and the RQ2
learning-curve data, and writes a single Markdown document with:
  1. a one-glance scorecard (one row per model);
  2. the full per-task benchmark tables (model x dataset, best-in-column bold);
  3. the cost table (train / tuning time, inference latency);
  4. the RQ2 crossover + small-data tables.

Output: notebooks/TABELA_RESULTADOS.md

Usage:
    uv run python scripts/build_results_table.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "TABELA_RESULTADOS.md"

PRIM = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
        "regression": ("rmse", False)}

# family + implementation provenance (from the presentation deck, section 1)
META = {
    "xgboost":        ("GBDT",       "oficial"),
    "lightgbm":       ("GBDT",       "oficial"),
    "catboost":       ("GBDT",       "oficial"),
    "tabpfn":         ("Foundation", "oficial"),
    "realmlp":        ("DL (MLP)",   "oficial"),
    "tabm":           ("DL (MLP)",   "oficial"),
    "tabnet":         ("DL (attn)",  "comunidade"),
    "mlp":            ("DL (MLP)",   "própria"),
    "ft_transformer": ("DL (attn)",  "própria"),
    "saint":          ("DL (attn)",  "própria"),
    "stab":           ("DL (attn)",  "própria"),
    "kan":            ("DL (KAN)",   "vendorizado"),
    "tabkan":         ("DL (KAN)",   "própria"),
    "tabfm":          ("Foundation", "oficial"),
}
# nicer display names
DISP = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "tabpfn": "TabPFN", "realmlp": "RealMLP", "tabm": "TabM", "tabnet": "TabNet",
    "mlp": "MLP", "ft_transformer": "FT-Transformer", "saint": "SAINT", "stab": "STab",
    "kan": "KAN", "tabkan": "TabKAN", "tabfm": "TabFM",
}


def fmt(v, nd):
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def task_table(t, task, nd):
    """model x dataset pivot of the primary metric, best-in-column bold + rank."""
    col, higher = PRIM[task]
    sub = t[t.task_type == task]
    piv = sub.pivot(index="model", columns="dataset", values=col)
    rank = piv.rank(ascending=not higher)
    mean_rank = rank.mean(axis=1)
    order = mean_rank.sort_values().index  # best first
    best = piv.max() if higher else piv.min()

    datasets = list(piv.columns)
    head = "| Modelo | " + " | ".join(datasets) + " | **Rank méd.** |"
    sep = "|" + "---|" * (len(datasets) + 2)
    lines = [head, sep]
    for m in order:
        cells = []
        for d in datasets:
            v = piv.loc[m, d]
            s = fmt(v, nd)
            if not pd.isna(v) and abs(v - best[d]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {DISP[m]} | " + " | ".join(cells) +
                     f" | {mean_rank[m]:.2f} |")
    return "\n".join(lines)


def main():
    t = pd.read_csv(REPO / "results" / "aggregated" / "test_results.csv")
    lat = pd.read_csv(REPO / "results" / "latency" / "latency_adult.csv"
                      ).set_index("model")["us_per_row"]

    # ----- per-task mean ranks (for the scorecard) -----
    ranks = {}
    for task, (col, higher) in PRIM.items():
        piv = t[t.task_type == task].pivot(index="model", columns="dataset", values=col)
        ranks[task] = piv.rank(ascending=not higher).mean(axis=1)
    rank_df = pd.DataFrame(ranks)

    # ----- cost -----
    cost = t.groupby("model").agg(train_med=("train_time_s", "median"),
                                  tune_med=("tuning_time_s", "median"))
    cost["latency_us"] = lat

    # ----- RQ2: small-data rank + crossover -----
    lc = pd.read_csv(REPO / "results" / "learning_curve" / "curve.csv")
    lc["primary"] = lc.apply(lambda r: r[PRIM[r.task_type][0]], axis=1)
    sm = lc[lc.train_size <= 4000]
    piv = sm.groupby(["dataset", "model"])["primary"].mean().reset_index().pivot(
        index="dataset", columns="model", values="primary")
    rk = piv.copy()
    for ds in rk.index:
        higher = PRIM[lc[lc.dataset == ds].task_type.iloc[0]][1]
        rk.loc[ds] = piv.loc[ds].rank(ascending=not higher)
    small_rank = rk.mean()

    def env(a, fam, higher):
        f = a[a.family == fam].groupby("train_size")["primary"]
        return f.max() if higher else f.min()

    cross_rows = []
    for ds in ["give_me_some_credit", "higgs", "adult", "jannis", "year_prediction"]:
        s = lc[lc.dataset == ds]; task = s.task_type.iloc[0]; higher = PRIM[task][1]
        a = s.groupby(["model", "family", "train_size"])["primary"].mean().reset_index()
        gb, dl = env(a, "gbdt", higher), env(a, "deep_learning", higher)
        sizes = sorted(set(gb.index) & set(dl.index)); gb, dl = gb.reindex(sizes), dl.reindex(sizes)
        lead = (gb - dl) if higher else (dl - gb)
        cross = next((sz for i, sz in enumerate(sizes)
                      if lead.iloc[i] > 0 and (lead.iloc[i:] > 0).all()), None)
        cross_rows.append((ds, task,
                           "GBDT" if lead.iloc[0] > 0 else "DL",
                           "GBDT" if lead.iloc[-1] > 0 else "DL",
                           str(cross) if cross is not None else "nunca (DL à frente)"))

    # ================= build the document =================
    md = []
    md.append("# Tabela mestra de resultados — benchmark tabular 2026\n")
    md.append("> **14 modelos × 18 datasets** (10 binárias · 3 multiclasse · 5 regressão), "
              "resultado no hold-out (seed 42). Gerado por `scripts/build_results_table.py` "
              "a partir de `results/aggregated/test_results.csv` (+ latência e curvas RQ2). "
              "Cobertura: **250/252** — as 2 ausências são exclusões estruturais no helena "
              "(100 classes excede o limite de TabPFN e TabFM). Latência e curvas RQ2 cobrem "
              "os 11 modelos originais (medições dos 3 novos: pendência registrada).\n")
    md.append("Métrica primária por tarefa: **binária → ROC-AUC (↑)**, "
              "**multiclasse → log-loss (↓)**, **regressão → RMSE (↓)**. "
              "Em cada coluna o **melhor valor está em negrito**; `Rank méd.` é o rank médio "
              "do modelo naquela tarefa (1 = melhor).\n")

    # 1. scorecard
    md.append("\n---\n\n## 1. Scorecard — visão de relance (1 linha por modelo)\n")
    md.append("Ranks médios por tarefa (↓ melhor) + custo típico. `Rank n≤4k` é o rank no "
              "regime de poucos dados (RQ2, só os 6 modelos varridos).\n")
    md.append("| Modelo | Família | Impl. | Rank bin. | Rank multi | Rank reg. | "
              "Treino (s) | Tuning (s) | Latência (µs/linha) | Rank n≤4k |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    scorecard_order = (rank_df["binary"]).sort_values().index
    for m in scorecard_order:
        fam, impl = META[m]
        rb = fmt(rank_df.loc[m, "binary"], 1)
        rm = fmt(rank_df.loc[m, "multiclass"], 1)
        rr = fmt(rank_df.loc[m, "regression"], 1)
        tr = fmt(cost.loc[m, "train_med"], 1)
        tu = fmt(cost.loc[m, "tune_med"], 0)
        la = fmt(cost.loc[m, "latency_us"], 1)
        sd = fmt(small_rank.get(m), 1)
        md.append(f"| **{DISP[m]}** | {fam} | {impl} | {rb} | {rm} | {rr} | "
                  f"{tr} | {tu} | {la} | {sd} |")
    md.append("\n*Leitura rápida: **TabFM** lidera as três tarefas (zero-shot), com latência "
              "proibitiva (~100 ms/linha em contextos grandes); **TabPFN** é o vice consistente; os "
              "**GBDTs** são os all-rounders baratos; **KAN/TabKAN** ficam no fundo junto ao TabNet "
              "(teste independente desfavorável); o DL pesado custa minutos de treino sem liderar.*")

    # 2. per-task tables
    md.append("\n---\n\n## 2. Binária — ROC-AUC ↑ (10 datasets)\n")
    md.append(task_table(t, "binary", 4))
    md.append("\n---\n\n## 3. Multiclasse — log-loss ↓ (3 datasets)\n")
    md.append("*TabPFN não roda o helena (100 classes > limite); rank médio sobre os 2 datasets restantes.*\n")
    md.append(task_table(t, "multiclass", 4))
    md.append("\n---\n\n## 4. Regressão — RMSE ↓ (5 datasets)\n")
    md.append("*RMSE está na escala de cada alvo — compare **dentro** da coluna, não entre colunas.*\n")
    md.append(task_table(t, "regression", 4))

    # 3. cost
    md.append("\n---\n\n## 5. Custo — treino, tuning e latência\n")
    md.append("Medianas entre datasets; latência medida no `adult`. **~300× de variação no treino, "
              "~26.000× na latência** — e o ranking de custo é quase o inverso da intuição.\n")
    md.append("| Modelo | Família | Treino final (s) | Tuning total (s) | Latência (µs/linha) |")
    md.append("|---|---|---|---|---|")
    for m in cost.sort_values("latency_us").index:
        fam, _ = META[m]
        md.append(f"| {DISP[m]} | {fam} | {fmt(cost.loc[m,'train_med'],2)} | "
                  f"{fmt(cost.loc[m,'tune_med'],0)} | {fmt(cost.loc[m,'latency_us'],2)} |")
    md.append("\n*A inversão do TabPFN: ~1s de treino e **zero tuning**, mas ~7.400 µs/linha "
              "de inferência (~22.000× o XGBoost). O tier rápido de latência inclui os 3 GBDTs "
              "**e** os DL tipo-MLP (RealMLP/MLP/TabM); só atenção, in-context e sampling são lentos.*")

    # 4. RQ2
    md.append("\n---\n\n## 6. RQ2 — curvas de aprendizado / crossover DL↔GBDT\n")
    md.append("Onde a liderança muda de mãos conforme o pool cresce (envelope best-of-família, "
              "500 → pool completo, 3 sementes). Ver figura `learning_curves.png` e o notebook "
              "`10_learning_curves.ipynb`.\n")
    md.append("| Dataset | Tarefa | Vence com pouco dado | Vence com muito dado | GBDT reassume em n≈ |")
    md.append("|---|---|---|---|---|")
    for ds, task, small, large, cross in cross_rows:
        md.append(f"| {ds} | {task} | {small} | {large} | {cross} |")
    md.append("\n**Rank no regime de poucos dados (n ≤ 4000), 1 = melhor:**\n")
    md.append("| Modelo | Rank médio n≤4k |")
    md.append("|---|---|")
    for m, v in small_rank.sort_values().items():
        md.append(f"| {DISP.get(m, m)} | {v:.2f} |")
    md.append("\n*Em 3 de 5 datasets o DL ultrapassa os GBDTs e não devolve a liderança; "
              "o TabPFN é o melhor em **todos** os datasets no regime de poucos dados.*")

    md.append("\n---\n\n*Para regenerar: `uv run python scripts/build_results_table.py`.*\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
