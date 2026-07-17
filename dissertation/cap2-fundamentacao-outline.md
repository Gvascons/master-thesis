# Capítulo 2 — Fundamentação Teórica (esqueleto v1)

> Estrutura com fontes mapeadas. Prosa a redigir em agosto (cronograma).
> Toda referência abaixo já foi verificada nos memorandos de `docs/` ou nos
> papers lidos durante a fase de integração; nada é citado de memória.

## 2.1 Dados tabulares e o problema da aprendizagem supervisionada
- Definição, heterogeneidade de features (numéricas/categóricas), ausência
  de estrutura espacial/sequencial — por que arquiteturas de visão/NLP não
  transferem trivialmente.
- Os três tipos de tarefa do benchmark.

## 2.2 Gradient boosting em árvores
- Boosting (Friedman 2001), XGBoost, LightGBM, CatBoost (tratamento nativo
  de categóricas / ordered boosting).
- Por que árvores casam com dados tabulares (viés indutivo de partições
  alinhadas a eixos; robustez a escala e a features irrelevantes).

## 2.3 Aprendizado profundo tabular
- MLPs e melhorias de receita: RealMLP (Holzmüller et al. 2024), TabM
  (Gorishniy et al. 2025 — ensembling eficiente por parâmetro).
- Atenção: TabNet, FT-Transformer, SAINT (atenção linha/coluna), STab.
- **Kolmogorov-Arnold Networks**: teorema de representação, ativações
  aprendíveis nas arestas (Liu et al. 2024, ICLR 2025); variantes de base
  (ChebyKAN etc.); TabKAN (Eslamian et al. 2025); a controvérsia empírica
  (Yu et al. 2024 "fairer comparison"; Poeta et al. 2024) — fonte:
  `docs/memo-novos-modelos-kan-tabkan-tabfm.md`.
- O gargalo do encoding categórico (one-hot, cardinalidade) — motivação
  observada no próprio benchmark (amazon_employee).

## 2.4 Foundation models tabulares e in-context learning
- Prior-fitted networks: pré-treino em datasets sintéticos, inferência como
  ICL em um forward (TabPFN v1/v2/v2.5).
- TabFM (Google 2026): arquitetura híbrida (atenção de colunas + compressão
  de linha + transformer ICL), checkpoints separados por tarefa, limites
  (10 classes, contexto) — fonte: memo de viabilidade + model card.
- O trade-off constitutivo: treino ~zero, inferência cara (o contexto viaja
  com o modelo).

## 2.5 Destilação de conhecimento
- Formulação clássica (Hinton et al. 2015; Bucila/Caruana 2006 — model
  compression e transfer sets).
- Destilação para árvores (born-again trees; Vidal & Schiffer 2020).
- Destilação de FMs tabulares: Pocket FM (2026) e o estado da questão —
  fonte: `docs/memo-novidade-destilacao.md`.
- Regressão distribucional: quantis, pinball loss, CRPS, calibração de
  intervalos (base para o Capítulo 7).

## 2.6 Comparação estatística de algoritmos
- Demšar 2006 (Friedman/Nemenyi/CD), correção de Iman-Davenport,
  Wilcoxon/Holm, Benavoli et al. 2017 (bayesiano/ROPE) — o Capítulo 4
  especifica o uso; aqui, a teoria mínima.
