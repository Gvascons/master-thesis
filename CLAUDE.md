# CLAUDE.md — convenções do projeto

Benchmark de mestrado: 14 modelos tabulares × 18 datasets OpenML + framework
de decisão + contribuição de destilação. Documento-mestre do programa:
`docs/programa-de-pesquisa.md`. Estado dos resultados: 250/252 células
(exclusões estruturais: TabPFN e TabFM × helena, 100 classes > limite).

## Regras operacionais (não negociáveis)

- **Python sempre via `uv run python`** (nunca `python` direto).
- **Commits: conta pessoal** (gvasconsbr@gmail.com / Gvascons), nunca a
  conta de trabalho. Push para `github.com/Gvascons/master-thesis`.
- **`agent-harness/` NUNCA entra em commits.**
- `results/aggregated/` é gitignored por padrão; CSVs pequenos de análise
  são adicionados com `git add -f` (o `fold_results.csv` de ~300MB fica
  local — reproduzível de `results/raw` + seed).
- Jobs longos: `setsid ... & disown` + log em `results/logs/` + marcador de
  saída; runners são resumíveis (pulam resultados existentes).
- Correções de registro via commit de retificação — nunca reescrever
  histórico já pushado.

## Contrato de conduta científica (resumo; íntegra no programa §7)

- Nenhuma afirmação de novidade sem verificação online datada.
- Hipóteses e critérios de decisão pré-registrados antes dos experimentos
  (`docs/desenho-experimental-destilacao.md`); mudanças viram adendo datado.
- Resultados negativos são reportados; smoke ≠ piloto ≠ resultado.
- Todo número citado tem origem em artefato versionado.

## Mapa rápido

| O quê | Onde |
|---|---|
| Protocolo/metodologia estatística | `docs/metodologia-estatistica.md` |
| Modelos (wrappers BaseModel) | `src/models/*.py` (template DL: `mlp_model.py`) |
| Espaços de busca Optuna | `configs/models.yaml` |
| Runner do benchmark | `scripts/run_all.py` (resumível; agrega ao final) |
| Harness de destilação (AI-2) | `scripts/distill.py` (estágios oof/students) |
| Validação LODO do framework | `scripts/lodo_validation.py` |
| Tabela mestra de resultados | `scripts/build_results_table.py` → `notebooks/TABELA_RESULTADOS.md` |
| Deck de apresentação | `notebooks/00_presentation.ipynb` (autocontido) |
| Rascunhos da dissertação | `dissertation/` (Markdown; conversão via pandoc depois) |
| Preprint (AI-2) | `paper/draft.md` (+ `outline.md`); figuras `pareto_distill` e `calibration_distill` |
| Análises AI-2 | `scripts/{pareto_strategies,ablation_*,mlp_student,extension_ctr23,paper_analysis,plot_*}.py` |
| Errata diamonds/kin8nm | `docs/errata-diamonds-kin8nm.md` (registro canônico) |
| Entregas em LaTeX (Overleaf) | `latex/{ai1,ai2}/` — artigos autocontidos; pendências no `latex/README.md` |

## Hardware

RTX 5080 (16 GB VRAM) + 32 GB RAM, GPU única. TabFM: backend PyTorch
(nunca JAX — 17-23GB), bf16, pesos em ~/.cache/huggingface (13 GB).
