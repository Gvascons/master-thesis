# latex/ — os dois trabalhos intermediários em formato artigo (Overleaf)

Dois projetos autocontidos, prontos para subir no Overleaf (New Project →
Upload Project → zip da pasta, ou copiar os arquivos):

| Pasta | Trabalho | Idioma | Fonte da verdade |
|---|---|---|---|
| `ai1/` | Trabalho Individual em Inteligência Computacional 1 — o benchmark 14×18 + arcabouço validado | inglês | `dissertation/cap5-benchmark.md`, `cap6-framework.md`, `notebooks/TABELA_RESULTADOS.md` |
| `ai2/` | Atividade de Orientação Individual — destilação distribucional + fronteira de Pareto | inglês | `paper/draft.md` (auditado 2×; este LaTeX também serve de base para o arXiv em out/2026) |

Idioma: inglês nos dois, por preferência registrada em 24/07/2026.

Cada projeto tem `main.tex` + `refs.bib` + `figures/*.pdf` (copiados de
`results/figures/`). Compilação: pdfLaTeX padrão do Overleaf (pdflatex →
bibtex → pdflatex ×2). Todos os números carregam comentários LaTeX com o
caminho do artefato de origem.

## Pendências antes da entrega (visíveis nos próprios arquivos)

1. **E-mail do orientador** nos dois `main.tex` (placeholder
   `[advisor email]`); o do aluno já está preenchido.
2. **Formato oficial** da entrega (a confirmar com a secretaria): se houver
   template obrigatório (ex.: SBC/ABNT/CIn), portar o conteúdo — a prosa e
   as tabelas transferem direto.

Referências: todas as entradas 2025-26 dos dois `refs.bib` foram conferidas
em 24/07/2026 contra as páginas primárias (arXiv, Springer, OpenReview,
ICLR, página de aceites do workshop FMSD). Correções aplicadas na conferência:
títulos reais de 4 trabalhos que eram citados pelo acrônimo do método
(TACO, TL-ANDI) ou por título presumido; listas de autores completas;
duas afirmações de prosa ajustadas ao que as fontes confirmam (TACO: "até
94× mais rápido" em vez de "~1% de contexto"; TL-ANDI: "anchoring +
distillation" em vez de "optimal transport").
