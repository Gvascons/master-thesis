# Guia de escrita didática — padrões verificados (05/10/2026)

> Extraído de 4 teses exemplares lidas em PDF (Grinsztajn/Saclay 2024 —
> o tema mais próximo do nosso; Feurer/Freiburg 2022; Gijsbers/TU-e 2022,
> fonte LaTeX pública em github.com/PGijsbers/Thesis; Hutter/UBC 2009,
> CAIAC Award) + Quadrio/Bicocca para fundamentação de transformers.
> PDFs preservados em `~/Desktop/masters/referencias-escrita/`.
> Complementa `docs/blueprint-dissertacao.md` (estrutura CIn). Guia
> vinculante para a escrita em `dissertacao/`.

## Regras de ouro (aplicar em TODO capítulo)

1. **Concreto antes de abstrato.** A 1ª página da introdução contém um
   dataset real, um número concreto ou uma figura legível por leigo
   (candidata: grid "mesmo dataset, GBDT vs MLP vs TabPFN" à la Gijsbers
   Fig. 1.1). Datasets nomeados em bullets de 2 frases ainda no cap. 1
   (Grinsztajn p. 9).
2. **Figuras: announce–show–interpret.** (i) frase de anúncio com
   propósito; (ii) legenda autocontida que ensina a ler (1ª frase em
   negrito = título-takeaway, Grinsztajn Fig. 2.2); (iii) interpretação
   imediata com achado + ressalva na mesma frase (Gijsbers p. 94). Cada
   figura nova é justificada pelo que a anterior não mostra. Setas
   "better" dentro de gráficos de trade-off (Grinsztajn Figs. 3.2-3.3).
3. **Tabelas:** no corpo, só agregadas, com linhas-resumo (rank médio,
   vitórias, p-values — Feurer Table 9) e convenção declarada na legenda
   (negrito = melhor; sublinhado = sem diferença significativa). A
   tabela mestra 14×18 vai para apêndice com zebra striping.
4. **Capítulos:** abertura com caixa-resumo + "Publicação associada" +
   parágrafo-roteiro; fechamento com conclusão do capítulo + caixa de
   transição plantando o próximo (Grinsztajn pp. 31/43/49). Macros em
   `dissertacao/thesisextras.sty`.
5. **Resultados negativos:** template "Surpreendentemente → investigamos
   → atribuímos a → resumo honesto" (Feurer); autocrítica datada "em
   retrospecto" (Grinsztajn §4.4); apêndice "cemitério de projetos
   abandonados" com rubricas (Grinsztajn Ap. I) — casa com nosso
   contrato de conduta (smoke ≠ piloto ≠ resultado).
6. **RQs híbridas:** QP1-QP3 formais no cap. 1 com diagrama QP→capítulo
   (Gijsbers Fig. 1.2); achados como títulos de subseção nos resultados
   ("Finding 1: ..." — Grinsztajn); ablações tituladas como perguntas
   ("Precisamos de OOF?" — Feurer).

## Capítulo de benchmark (template Gijsbers cap. 5)

Ordem: abertura (problema + contribuição com números + roteiro) →
related que justifica o design (defeitos metodológicos dos anteriores,
com cortesia explícita) → seleção de modelos com exclusões nominais
justificadas + baselines em subseção → design separado de resultados
(critérios de datasets em bullets rubricados; política de células
faltantes como decisão estatística defendida; hardware com justificativa)
→ **limitações do design ANTES dos resultados** (incl. contaminação:
pré-treino dos TFMs sobre OpenML) → resultados em camadas de agregação,
cada uma compensando a anterior (CD → boxplots escalonados → análise
condicionada → Pareto) → falhas como subseção com taxonomia → conclusão
do capítulo com números + ressalva.

## Fundamentação (template Grinsztajn §§1.2-1.3 + Quadrio)

Sequência: dataframe real impresso → árvore única com exemplo guiado →
boosting VERBAL via viés-variância (zero fórmulas; rodapé técnico) →
MLP com 2 equações → atenção com equações completas + diagrama do
transformer com decoder ACINZENTADO (Quadrio Fig. 2.1) → ICL/TabPFN pelo
mecanismo do forward pass (Nagler 2023 citado sem derivar) → destilação
com matemática completa (é contribuição). **Profundidade matemática
proporcional à proximidade da contribuição.** Figuras consagradas podem
ser emprestadas com crédito, não redesenhadas. Rubricas em negrito a
cada 1-2 parágrafos; títulos-pergunta onde couber.

## Checklist de riqueza (adotados)

Caixa-resumo de capítulo; caixa de transição; nota de proveniência
capítulo ↔ latex/ai1-ai2; "Como ler esta dissertação" (meia página);
diagrama QP→capítulo; diagrama de pipeline com código de cores explicado;
apêndice de reprodutibilidade (links + seeds + versões + passo-a-passo);
delegação de artefatos grandes ao repositório (fold_results.csv);
cemitério de abandonados; conclusão com "conselhos ao praticante" em
checklist (= nosso framework, à la Hutter §14.1) e adendo datado de
impacto/autocrítica (Feurer cap. 9); resumo/abstract bilíngue estruturado
nas contribuições.
