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

## Pendências antes da entrega

1. **Formato oficial** da entrega (a confirmar com a secretaria): se houver
   template obrigatório (ex.: SBC/ABNT/CIn), portar o conteúdo — a prosa e
   as tabelas transferem direto. Não há norma pública do CIn para entregas
   de disciplina (verificado 04/10/2026); formato de artigo com `abbrvnat`
   é defensável, mas confirme. (E-mails dos dois autores: preenchidos.)

**Auditoria round 4 (05/10/2026, final pré-entrega):** 3 releitores
integrais + verificador web + compilação local determinística (tectonic)
com inspeção página a página dos PDFs. Correções: 2 edits perdidos por
scripts abortados restaurados (reconciliação §5.1 AI-2; claim full-slate
§4.4 AI-1); claim "sub-ms ⇒ só students" reescopada em 6 pontos (kin8nm é
contraexemplo na própria figura — virou ilustração da regra de decisão);
figura de Pareto do AI-1 refeita em rank médio × tempo mediano (a versão
AUC contradizia as fronteiras do texto); terminologia "11-model core
slate"; xcolor adicionado (erro engolido pelo Overleaf); ~30 polimentos de
costura/gramática/dêixis; 6 citações novas verificadas (conformal,
quantização FP8, GEAR, 2º benchmark CDE, meta-features, Poeta/Rice no
AI-1); claims datadas 100% validadas contra fontes primárias em 05/10.

**Auditoria round 3 (04/10/2026):** 5 agentes (números AI-1, números AI-2,
referências, concorrência/banca, estrutura/didática). Resultado: espinha
numérica íntegra (AI-1 ~60 números recomputados; AI-2 Tabela 2 rederivada
dos CSVs brutos); correções aplicadas — claim das KANs reescopada (Poeta
et al. 2024 antecede), título AI-2 Cache→Shrink, reconciliação com arXiv
2610.01435, TR oficial do TabFM citado (2609.37959), caveat de retenção
com denominador pequeno + colunas n/ΔCRPS, figuras AI-2 regeneradas em
inglês, legendas corrigidas (painéis RMSE/CRPS; CD/Pareto do AI-1
escopados), tabela de 18 datasets no AI-1, diagrama do pipeline no AI-2,
glosses de CRPS/PICP/ROPE/η²/ICL, protocolo de latência explicitado
(dispositivo/regime). Adendos datados nos memos de novidade/modelos.

Referências: conferidas em 24/07/2026 e **re-conferidas em 04/10/2026**
(39 entradas; zero inexistentes; atualizações de status aplicadas:
McElfresh +2 autores, MotherNet título v2, Prior Labs autores reais,
TabFM TR oficial) contra as páginas primárias (arXiv, Springer, OpenReview,
ICLR, página de aceites do workshop FMSD). Correções aplicadas na conferência:
títulos reais de 4 trabalhos que eram citados pelo acrônimo do método
(TACO, TL-ANDI) ou por título presumido; listas de autores completas;
duas afirmações de prosa ajustadas ao que as fontes confirmam (TACO: "até
94× mais rápido" em vez de "~1% de contexto"; TL-ANDI: "anchoring +
distillation" em vez de "optimal transport").
