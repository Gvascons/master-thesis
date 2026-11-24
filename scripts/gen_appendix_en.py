#!/usr/bin/env python3
"""Gera as versões EN dos arquivos de tabelas geradas dos apêndices.

Lê os arquivos GERADOS da versão PT (dissertacao/appendix/*.tex), traduz
as strings fixas (captions, cabeçalhos) por pares exatos e converte o
formato numérico (separador decimal {,} -> . e, no bloco da tabela de
custos, o formato PT sem chaves). Nenhum valor numérico é alterado,
apenas o formato. Checagens reais ao final: (i) cada par de tradução
precisa ter sido aplicado em ao menos um arquivo; (ii) nenhuma célula
ou legenda pode restar em formato numérico PT (vírgula decimal nua ou
híbrido milhar+decimal). Saída: dissertacao-en/appendix/.

Uso: uv run python scripts/gen_appendix_en.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dissertacao" / "appendix"
DST = ROOT / "dissertacao-en" / "appendix"

# pares (pt, en, obrigatório em qual arquivo)
PAIRS = [
    # --- tabelas-geradas.tex ---
    ("Classificação binária --- resultados completos",
     "Binary classification --- complete results"),
    ("Classificação binária --- AUC-ROC no teste (maior é melhor)",
     "Binary classification --- test AUC-ROC (higher is better)"),
    ("Classificação multiclasse --- resultados completos",
     "Multiclass classification --- complete results"),
    ("Classificação multiclasse --- log-loss no teste (menor é melhor)",
     "Multiclass classification --- test log-loss (lower is better)"),
    ("Regressão --- resultados completos",
     "Regression --- complete results"),
    ("Regressão --- RMSE no teste (menor é melhor)",
     "Regression --- test RMSE (lower is better)"),
    ("Métrica primária por modelo em cada \\emph{dataset}; "
     "\\textbf{negrito} = melhor da linha; ``---'' = célula "
     "estruturalmente ausente.",
     "Primary metric per model on each dataset; \\textbf{bold} = best "
     "in row; ``---'' = structurally absent cell."),
    ("Métricas secundárias --- tarefa binária",
     "Secondary metrics --- binary task"),
    ("Métricas secundárias --- tarefa multiclasse",
     "Secondary metrics --- multiclass task"),
    ("Métricas secundárias --- tarefa regressão",
     "Secondary metrics --- regression task"),
    ("Rank médio (1 = melhor) dos 14 modelos em cada métrica da tarefa "
     "binária; a primária aparece como referência. Valores por célula em",
     "Mean rank (1 = best) of the 14 models on each metric of the binary "
     "task; the primary metric appears for reference. Per-cell values in"),
    ("Rank médio (1 = melhor) dos 14 modelos em cada métrica da tarefa "
     "multiclasse; a primária aparece como referência. Valores por célula em",
     "Mean rank (1 = best) of the 14 models on each metric of the "
     "multiclass task; the primary metric appears for reference. "
     "Per-cell values in"),
    ("Rank médio (1 = melhor) dos 14 modelos em cada métrica da tarefa "
     "regressão; a primária aparece como referência. Valores por célula em",
     "Mean rank (1 = best) of the 14 models on each metric of the "
     "regression task; the primary metric appears for reference. "
     "Per-cell values in"),
    ("Estatística KS por conjunto --- binária",
     "KS statistic per dataset --- binary"),
    ("Estatística de Kolmogorov--Smirnov (KS; maior é melhor) por modelo "
     "em cada conjunto binário --- a métrica de separação padrão da "
     "indústria de crédito (Seção~\\ref{sec:met-metricas}); "
     "\\textbf{negrito} = melhor da linha.",
     "Kolmogorov--Smirnov statistic (KS; higher is better) per model on "
     "each binary dataset --- the credit industry's standard separation "
     "metric (Section~\\ref{sec:met-metricas}); \\textbf{bold} = best "
     "in row."),
    ("Tamanhos de efeito ($d$ de Cohen) --- binária",
     "Effect sizes (Cohen's $d$) --- binary"),
    ("$d$ de Cohen pareado (linha $-$ coluna) na tarefa binária "
     "(AUC-ROC); positivo = linha melhor (AUC maior).",
     "Paired Cohen's $d$ (row $-$ column) on the binary task (AUC-ROC); "
     "positive = row better (higher AUC)."),
    ("Tamanhos de efeito ($d$ de Cohen) --- regressão",
     "Effect sizes (Cohen's $d$) --- regression"),
    ("$d$ de\nCohen pareado (linha $-$ coluna) na tarefa regressão "
     "(RMSE); positivo\n= linha \\emph{pior} (RMSE maior). Ressalva: o "
     "$d$ é computado sobre\nRMSEs brutos de 5 conjuntos em escalas de "
     "alvo diferentes, e as\ndiferenças são dominadas pelos conjuntos "
     "de maior escala; a leitura é\nindicativa, e as comparações "
     "formais do capítulo usam ranks.",
     "Paired Cohen's $d$ (row $-$ column) on the regression task "
     "(RMSE); positive = row \\emph{worse} (higher RMSE). Caveat: the "
     "$d$ is computed on raw RMSEs of 5 datasets with different target "
     "scales, and the differences are dominated by the larger-scale "
     "datasets; the reading is indicative, and the chapter's formal "
     "comparisons use ranks."),
    ("IC bootstrap da média de AUC --- binária",
     "Bootstrap CI of the mean AUC --- binary"),
    ("Média de AUC-ROC na tarefa binária com IC95\\% de \\emph{bootstrap} "
     "percentil ($B{=}10\\,000$) sobre os conjuntos.",
     "Mean AUC-ROC on the binary task with a percentile-bootstrap 95\\% "
     "CI ($B{=}10\\,000$) over the datasets."),
    ("Custos consolidados por modelo", "Consolidated costs per model"),
    ("Custos medianos por modelo: treino (s, 18 conjuntos), tuning (s, "
     "excl.\\ \\texttt{adult} --- Seção~\\ref{sec:met-custo}) e latência "
     "(\\textmu s/linha, \\texttt{adult}).",
     "Median costs per model: training (s, 18 datasets), tuning (s, "
     "excl.\\ \\texttt{adult} --- Section~\\ref{sec:met-custo}), and "
     "latency (\\textmu s/row, \\texttt{adult})."),
    ("Variabilidade entre \\emph{folds}", "Across-fold variability"),
    ("AUC-ROC", "ROC-AUC"),
    ("Mean rank (1 = best)", "Average rank (1 = best)"),
    ("<conjunto>", "<dataset>"),
    ("Mediana, sobre os conjuntos da tarefa, do coeficiente de variação "
     "entre \\emph{folds} da métrica primária (\\%; menor = mais "
     "estável).",
     "Median, over the task's datasets, of the across-fold coefficient "
     "of variation of the primary metric (\\%; lower = more stable)."),
    ("\\emph{Conjunto}", "\\emph{Dataset}"),
    ("\\emph{Modelo}", "\\emph{Model}"),
    ("Acurácia balanceada", "Balanced accuracy"),
    ("& Acurácia &", "& Accuracy &"),
    ("F1-ponderado", "Weighted F1"),
    ("F1-macro", "Macro-F1"),
    ("& Média & IC95\\% \\\\", "& Mean & 95\\% CI \\\\"),
    ("Treino (s) & Tuning (s) & Latência (\\textmu s/linha)",
     "Training (s) & Tuning (s) & Latency (\\textmu s/row)"),
    ("Binária (AUC) & Multiclasse (log-loss) & Regressão (RMSE)",
     "Binary (AUC) & Multiclass (log-loss) & Regression (RMSE)"),
    # --- tabela-metricas-operacionais.tex ---
    ("Métricas operacionais no \\emph{hold-out} --- binárias",
     "Operational metrics on the hold-out --- binary"),
    ("Métricas operacionais no \\emph{hold-out} (limiar 0,5; classe de "
     "interesse = minoritária).",
     "Operational metrics on the hold-out (threshold 0.5; class of "
     "interest = minority)."),
    ("Precisão (min.)", "Precision (min.)"),
    ("Precisão macro", "Macro precision"),
    ("Recall macro", "Macro recall"),
    ("F1 macro", "Macro F1"),
    # --- hiperparametros-escolhidos.tex ---
    ("Hiperparâmetros escolhidos ---", "Chosen hyperparameters ---"),
    (": configuração vencedora do Optuna em cada conjunto.",
     ": the winning Optuna configuration on each dataset."),
    ("campo \\texttt{best\\_params}", "field \\texttt{best\\_params}"),
    # genéricos (por último)
    ("Fonte: idem.", "Source: idem."),
    ("Fonte:", "Source:"),
    ("ARQUIVO GERADO - NAO EDITAR A MAO.",
     "GENERATED FILE (EN variant) - DO NOT EDIT BY HAND."),
]

FILES = ["tabelas-geradas.tex", "tabela-metricas-operacionais.tex",
         "hiperparametros-escolhidos.tex"]


def convert_custos_block(s: str) -> str:
    """Converte o bloco da tabela de custos (tab:app-custos), que o
    gerador PT emite em formato PT *sem chaves* (vírgula decimal nua e
    ponto de milhar): 8.644,82 -> 8,644.82."""
    start = s.find(r"\label{tab:app-custos}")
    if start == -1:
        return s
    end = s.find(r"\end{table}", start)
    block = s[start:end]
    # 1) ponto de milhar -> placeholder; 2) vírgula decimal -> ponto;
    # 3) placeholder -> vírgula (ordem obrigatória)
    block = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "\x00", block)
    block = re.sub(r"(?<=\d),(?=\d)", ".", block)
    block = block.replace("\x00", ",")
    return s[:start] + block + s[end:]


def check_en_numbers(name: str, s: str) -> None:
    """Falha se restar número em formato PT nas linhas de célula de
    tabela: vírgula decimal nua (d,dd — teria pego 28 das 35 células do
    caso tab:app-custos) ou híbrido milhar+decimal PT (d.ddd,dd). O
    milhar PT isolado ("1.117") é sintaticamente indistinguível de um
    decimal EN e não é detectável aqui; o bloco de custos é convertido
    explicitamente por convert_custos_block."""
    bad = []
    for i, line in enumerate(s.splitlines(), 1):
        if "&" not in line and "caption" not in line:
            continue
        if re.search(r"\d,\d{1,2}(?!\d)(?!\})", line):
            bad.append((i, "virgula decimal PT", line.strip()[:60]))
        if re.search(r"\d{1,3}\.\d{3},\d", line):
            bad.append((i, "hibrido milhar+decimal PT", line.strip()[:60]))
        if re.search(r"\{=\}\d{1,3}\.\d{3}\$", line):
            bad.append((i, "milhar com ponto em expressao", line.strip()[:60]))
    assert not bad, f"{name}: formato PT residual: {bad[:5]}"


def main():
    DST.mkdir(exist_ok=True)
    applied = {pt: 0 for pt, _ in PAIRS}
    for name in FILES:
        s = (SRC / name).read_text()
        for pt, en in PAIRS:
            if pt in s:
                applied[pt] += s.count(pt)
            s = s.replace(pt, en)
        # separador decimal: 0{,}123 -> 0.123 (apenas entre dígitos)
        s, n = re.subn(r"(?<=\d)\{,\}(?=\d)", ".", s)
        s = convert_custos_block(s)
        # B=10\,000 (thin space PT) -> 10{,}000 — APOS o regex do decimal,
        # para nao virar 10.000 (erro do lote anterior)
        s = s.replace("$B{=}10\\,000$", "$B{=}10{,}000$")
        # intervalos mantem o padrao dos capitulos: "[a;\ b]"
        resid = re.findall(r"[çãõáéíóúâêô]", s)
        check_en_numbers(name, s)
        header = ("% Gerado por scripts/gen_appendix_en.py a partir de "
                  f"dissertacao/appendix/{name} (nao editar a mao).\n")
        (DST / name).write_text(header + s)
        print(f"{name}: {n} decimais convertidos; residuo acentos: "
              f"{len(resid)}; checagem de formato EN: ok")
    unused = [pt[:50] for pt, c in applied.items() if c == 0]
    assert not unused, f"pares nunca aplicados (desatualizados?): {unused}"
    print(f"pares aplicados: {sum(1 for c in applied.values() if c)}/"
          f"{len(applied)}")


if __name__ == "__main__":
    main()
