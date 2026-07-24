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
2. **Referências marcadas `TODO-verify`** nos dois `refs.bib`: entradas de
   2025-26 cujos IDs de arXiv vêm dos memorandos datados, mas cujas listas
   de autores/títulos exatos não foram registradas lá (regra do projeto:
   não inventar metadados). Verificar online antes da versão final:
   - ai2: pocketfm2026, pocketfmhealth2026, tabdistill2025, tfmgam2026,
     taco2026, crumb2026, tlandi2026, tabpfn3-2026, tabicl2025, tabflex2025
   - ai1: tabkan2025, kancritical2024, stab-ref, tabm2025
3. **Formato oficial** da entrega (a confirmar com a secretaria): se houver
   template obrigatório (ex.: SBC/ABNT/CIn), portar o conteúdo — a prosa e
   as tabelas transferem direto.
