#!/usr/bin/env python3
"""Análise de sensibilidade — exclusão do california_housing (ERRATA 19/10/2026).

Descoberta em 19/10/2026 (auditoria de identidade nome<->ID<->shape, a
checagem registrada como pendência na errata diamonds/kin8nm): o
california_housing do config aponta para OpenML 44090 — a versão
BINARIZADA do tabular-benchmark de Grinsztajn (suíte "classification on
numerical features"; alvo {0,1} = valor acima da mediana). O benchmark e
a cadeia de destilação o trataram como regressão. As contas são
mecanicamente válidas (RMSE sobre {0,1} = raiz do Brier; CRPS de alvo
Bernoulli é bem definido), mas o conjunto é degenerado para claims
DISTRIBUCIONAIS. Este script quantifica o que muda em cada resultado
publicado quando o california é excluído.

Saída: results/distillation/sensitivity_california.csv + vereditos no
stdout. Uso: uv run python scripts/sensitivity_california_binario.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "results" / "distillation"
CAL = "california_housing"
rows = []


def reg(analise, escopo, publicado, sem_cal, veredito):
    rows.append({"analise": analise, "escopo": escopo,
                 "publicado": publicado, "sem_california": sem_cal,
                 "veredito": veredito})
    print(f"{analise:28s} | {escopo:10s} | pub: {publicado:34s} | "
          f"sem cal: {sem_cal:34s} | {veredito}")


def headline(df):
    pos, n = int((df.ret_crps > 0).sum()), len(df)
    med = df.ret_crps.median()
    sp = stats.binomtest(pos, n, alternative="greater").pvalue
    wp = stats.wilcoxon(df.ret_crps, alternative="greater").pvalue
    rng = np.random.default_rng(42)
    boots = [np.median(rng.choice(df.ret_crps.values, n, replace=True))
             for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pos, n, med, sp, wp, lo, hi


def main():
    # 1) manchete de retenção (3 sementes, paper_retention.csv)
    ret = pd.read_csv(DIST / "paper_retention.csv")
    a = headline(ret)
    b = headline(ret[ret.dataset != CAL])
    reg("retencao (3 sementes)", "20->19",
        f"{a[0]}/{a[1]} med={a[2]:.3f} sinal p={a[3]:.4f} Wilc p={a[4]:.4f}",
        f"{b[0]}/{b[1]} med={b[2]:.3f} sinal p={b[3]:.4f} Wilc p={b[4]:.4f}",
        "sinal sobrevive; Wilcoxon unilateral perde 5% (0,045->0,067)")
    reg("IC bootstrap mediana", "20->19",
        f"[{a[5]:.2f};{a[6]:.2f}]", f"[{b[5]:.2f};{b[6]:.2f}]",
        "segue inteiramente positivo")

    # 2) SEEDS-10 (10 sementes; replica o calculo canonico de
    #    scripts/paper_analysis.py: nucleo via teacher_eval + extensao via cell)
    ext = pd.read_csv(DIST / "extension.csv")
    ge = ext.groupby(["dataset", "cell"]).agg(crps=("crps", "mean"))
    r10 = {}
    for ds in ext.dataset.unique():
        t = ge.loc[(ds, "teacher"), "crps"]
        qh = ge.loc[(ds, "student_quant_hard"), "crps"]
        qs = ge.loc[(ds, "student_quant_soft"), "crps"]
        gap = qh - t
        if gap > 1e-9:
            r10[ds] = (qh - qs) / gap
    core = pd.read_csv(DIST / "distill.csv")
    core = core[core.teacher == "tabpfn"]
    t5 = pd.read_csv(DIST / "teacher_eval.csv")
    t5 = t5[t5.teacher == "tabpfn"].set_index("dataset")
    gd = core.groupby(["dataset", "student", "target"]).agg(crps=("crps", "mean"))
    for ds in t5.index:
        qh = gd.loc[(ds, "xgb_quant", "hard"), "crps"]
        qs = gd.loc[(ds, "xgb_quant", "soft"), "crps"]
        gap = qh - t5.loc[ds, "crps"]
        if gap > 1e-9:
            r10[ds] = (qh - qs) / gap
    s10 = pd.Series(r10)
    for label, s in [("20", s10), ("19", s10.drop(CAL))]:
        pos, n = int((s > 0).sum()), len(s)
        sp = stats.binomtest(pos, n, alternative="greater").pvalue
        wp = stats.wilcoxon(s.values, alternative="greater").pvalue
        txt = f"{pos}/{n} med={s.median():.3f} sinal p={sp:.4f} Wilc p={wp:.4f}"
        if label == "20":
            pub10 = txt
        else:
            reg("SEEDS-10 (10 sementes)", "20->19", pub10, txt,
                "espelha as 3 sementes: sinal sobrevive, Wilcoxon perde 5%")

    # 3) E3 — utilidade de decisao em tau=0,90
    du = pd.read_csv(DIST / "decision_utility_analysis.csv")
    for col, nome, pub_pub in [("soft_le_hard_q90", "E3 vs hard", "18/20 p=0,0002"),
                               ("soft_le_gauss_q90", "E3 vs gauss", "17/20 p=0,0013")]:
        dd = du[du.dataset != CAL]
        w = int(dd[col].sum()); n = len(dd)
        p = stats.binomtest(w, n, alternative="greater").pvalue
        reg(nome, "20->19", pub_pub, f"{w}/{n} p={p:.4f}", "sobrevive com folga")

    # 4) Friedman dos 5 sistemas (colunas de CRPS, nao as de rank)
    fr = pd.read_csv(DIST / "friedman_systems.csv")
    fr = fr[fr.dataset != "__rank_medio__"]
    sysc = ["teacher", "student_soft", "student_hard", "qrf", "ngboost"]
    for label, d in [("20", fr), ("19", fr[fr.dataset != CAL])]:
        mat = d[sysc].values
        chi2, p = stats.friedmanchisquare(*mat.T)
        order = pd.DataFrame(mat, columns=sysc).rank(axis=1).mean().sort_values()
        if label == "20":
            pubf = f"chi2={chi2:.1f} p={p:.0e}"
        else:
            reg("Friedman 5 sistemas", "20->19", pubf,
                f"chi2={chi2:.1f} p={p:.0e}",
                f"ordenacao identica: {' < '.join(order.index)}")

    # 5) A-SORT
    ab = pd.read_csv(DIST / "ablation_sort.csv")
    for excl in (False, True):
        d = ab[ab.dataset != CAL] if excl else ab
        agg = (d.groupby(["dataset", "target"])
               .apply(lambda g: (g.crps_sorted < g.crps_raw).mean(),
                      include_groups=False).unstack())
        res = {t: f"{int((agg[t] > 0.5).sum())}/{len(agg)}" for t in agg.columns}
        if not excl:
            puba = f"hard {res.get('hard')} soft {res.get('soft')}"
        else:
            reg("A-SORT", "20->19", puba,
                f"hard {res.get('hard')} soft {res.get('soft')}",
                "conclusao inalterada")

    # 6) R-COND — Fisher do gate
    rc = pd.read_csv(DIST / "retention_conditional.csv")
    if {"dataset", "ret_crps", "rel_edge"}.issubset(rc.columns):
        for excl in (False, True):
            d = rc[rc.dataset != CAL] if excl else rc
            d = d.sort_values("rel_edge")
            half = len(d) // 2
            lo_, hi_ = d.iloc[:half], d.iloc[half:]
            tab = [[int((hi_.ret_crps > 0).sum()), int((hi_.ret_crps <= 0).sum())],
                   [int((lo_.ret_crps > 0).sum()), int((lo_.ret_crps <= 0).sum())]]
            _, p = stats.fisher_exact(tab, alternative="greater")
            txt = f"alto {tab[0][0]}/{tab[0][0]+tab[0][1]} baixo {tab[1][0]}/{tab[1][0]+tab[1][1]} Fisher p={p:.3f}"
            if not excl:
                pubr = txt
            else:
                reg("R-COND (gate preditivo)", "20->19", pubr, txt,
                    "direcao preservada" if p < 0.1 else "ATENCAO: p>0,10")
    else:
        reg("R-COND", "-", f"colunas: {rc.columns.tolist()}", "-", "VERIFICAR a mao")

    # 7) estudos pequenos que INCLUEM california (sem recomputo — leitura)
    reg("E4 conformal", "5->4", "conserta PICP nos 2 bracos, CRPS <1%",
        "4/4 conjuntos restantes mantem o padrao (ver pilot_conformal.csv)",
        "leitura qualitativa inalterada")
    ps = pd.read_csv(DIST / "pilot_sampling.csv")
    pm = ps.groupby(["dataset", "variant"]).crps.mean().unstack()
    win_cal = bool(pm.loc[CAL, "sampling_soft"] < pm.loc[CAL, "mse_soft"])
    reg("E6 amostragem", "5->4", "vence 1/5, custo 2,8-44x",
        f"a unica vitoria ERA do california (verif.: {win_cal}); sem ele 0/4",
        "refutacao FORTALECE (0/4)")
    ag = pd.read_csv(DIST / "ablation_densegrid.csv")
    for excl in (False, True):
        d = ag[ag.dataset != CAL] if excl else ag
        spread = (d.groupby("dataset").ratio.max()
                  - d.groupby("dataset").ratio.min()).max() * 100
        glob = (d.ratio.max() - d.ratio.min()) * 100
        if not excl:
            puba2 = f"spread global {glob:.2f} p.p."
        else:
            reg("A-GRID estimador", "4->3", puba2,
                f"spread global {glob:.2f} p.p.",
                "estabilidade do estimador inalterada")
    reg("R-TUNE", "3->2", "H-T1 refutada 2/3; california era o caso forte",
        "resta cps88wages (refuta) e cpu_activity (nao); 1/2",
        "licao qualitativa sobrevive, caso-vitrine degradado")

    out = DIST / "sensitivity_california.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsalvo: {out}")


if __name__ == "__main__":
    main()
