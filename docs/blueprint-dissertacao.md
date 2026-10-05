# Blueprint da dissertação — padrões CIn-UFPE verificados (05/10/2026)

> Levantamento em fontes primárias: 4 dissertações de mestrado do PPGCC
> lidas integralmente (3 orientadas por Germano C. Vasconcelos), regimento
> do PPGCC e manual de normalização SIB/UFPE. Guia vinculante para a
> escrita em `dissertacao/`.

## Exemplares analisados (PDFs no repositório Attena)

| Autor/ano | Orientador | Pp. | Traço estrutural a reusar |
|---|---|---|---|
| Gama Neto 2022 (handle 45794) | Germano | ~101 | **Dois capítulos de contribuição com resultados próprios** (comparativo + método novo) — espelha nossa divisão AI-1/AI-2; apêndices de dados |
| Souza Júnior 2022 (handle 50327) | Germano | 82 | Perfil benchmark (~25 tabelas); **testes não-paramétricos apresentados na Fundamentação** e aplicados nos Resultados; parágrafo-ponte abrindo capítulos |
| Costa, Esdras 2022 (handle 45616) | Germano | 84 | Seção **"Limitações e Ameaças à Validade" na Conclusão**; "Problema de Pesquisa" como seção da Introdução |
| Silva 2021 (handle 44642) | D. Cunha | 69 | Introdução com **"Resumo das contribuições" + "Produção bibliográfica"** |

Inglês é aceito na prática (Costa, Renan 2023, handle 54752), mas o
majoritário é português → **dissertação em PT** (consistente com os
rascunhos de `dissertation/*.md`).

## Normas (fontes primárias)

- Regimento PPGCC (art. 34): formato definido pelo Colegiado; sem previsão
  de "coleção de artigos" no acadêmico → **monográfico**. Art. 37: banca
  3-4 doutores, ≥1 externo. Art. 39: 90 dias p/ correções; depósito via
  SIGAA (ficha catalográfica gerada pela biblioteca — folha em branco no
  draft).
- Manual SIB/UFPE (08/2023): NBR 14724/10520:2023/6028/6027/6023:2018;
  resumo 150-500 palavras, 1 parágrafo; 3-6 palavras-chave com ponto e
  vírgula; paginação a partir da Introdução; seção primária em folha nova.
- Referências: **ABNT autor-data** (`abnt.bst` do template risethesis) —
  padrão dos 4 exemplares.

## Estrutura adotada (8 capítulos, ~110-130 pp)

1. **Introdução** — Contexto e motivação; Problema de pesquisa; Objetivos
   (geral + específicos); Perguntas de pesquisa QP1-QP3 (refinamento
   nosso, compatível); **Resumo das contribuições; Produção
   bibliográfica** (AI-1, AI-2/preprint); Estrutura da dissertação (um
   parágrafo por capítulo).
2. **Fundamentação teórica** — dados tabulares; GBDT; DL tabular; TFMs
   (TabPFN/TabFM/ICL); destilação; regressão distribucional e calibração;
   **metodologia estatística (Friedman/Nemenyi/CD/bayesiano-ROPE)** aqui
   (padrão Souza Júnior).
3. **Trabalhos relacionados** — capítulo próprio; fecha com tabela-síntese
   de lacunas.
4. **Metodologia experimental** — datasets, modelos, HPO, protocolo,
   métricas, hardware, reprodutibilidade/conduta.
5. **Benchmark multicritério (AI-1)** — 4 atos + análise estatística.
6. **Arcabouço de decisão validado** — LODO, políticas, **regret
   multicritério ponderado (E2)**, flowchart @14.
7. **Destilação distribucional (AI-2)** — pré-registro, 6 fases, fronteira,
   ablações, **baselines clássicos (E1)**, regra de decisão.
8. **Conclusão** — Considerações finais; Contribuições; **Limitações e
   ameaças à validade** (padrão Esdras; inclui exclusões estruturais e
   errata diamonds/kin8nm); Trabalhos futuros.
- Apêndices: A) tabelas completas por dataset; B) espaços de busca;
  C) suplementares/ablações.

## Padrões didáticos a aplicar

- Parágrafo-ponte abrindo todo capítulo ("Este capítulo...").
- Figuras/tabelas sempre anunciadas-mostradas-interpretadas no texto.
- Densidade alvo: 30-50 figuras + 15-25 tabelas (perfil misto
  benchmark+método).
- Acrônimos via `\listofacronyms` + expansão no primeiro uso.
- QP numeradas ecoadas nas seções de resultados e retomadas na conclusão.

## Esqueleto LaTeX

`dissertacao/` — risethesis (msc, pt, oneside), cls com patch documentado
(\quotefont vs TeX Live moderno); compila no tectonic. PDFs dos exemplares
baixados no scratchpad da sessão de 05/10 (não versionados).
