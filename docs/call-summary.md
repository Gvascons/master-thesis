# Acompanhamento Individual 1 — Resumo para a reunião

**João Gabriel** · benchmark de 11 modelos × 18 datasets (OpenML) · nested CV (5×3) ·
Optuna TPE · RTX 5080 · 197/198 experimentos (tabpfn×helena N/A: 100 classes > limite).

---

## A pergunta
Em 2025, com DL tabular moderno (TabM, RealMLP, STab) e foundation model (TabPFN v2),
o "GBDT sempre vence em dados tabulares" ainda vale — e **se a acurácia não separa mais
o campo, o que deveria guiar a escolha?**

## A resposta, em 4 atos
1. **O empate.** No desempenho bruto, o topo é estatisticamente indistinguível. Friedman
   acusa diferença (binária p=0.0005), mas o post-hoc não separa o cluster de topo
   (TabPFN, 3 GBDTs, TabM, RealMLP). Confirmado também por **teste Bayesiano (signed-rank
   + ROPE)**: TabPFN e XGBoost são **~94% praticamente equivalentes**. A variância de rank
   é **80% intra-família** (η²=0.20) → "GBDT vs DL" é uma abstração fraca; o modelo importa
   mais que a família. *Só o fundo é robusto: TabNet é o pior de forma consistente.*
2. **Os eixos ocultos.** O que separa o campo saiu da acurácia e foi para o custo: tempo
   de treino varia **~300×**, latência de inferência **~26.000×**. Achado-âncora — a
   **inversão do TabPFN**: o mais barato de treinar (~1 s, zero tuning) é o **2º mais caro
   de servir** (72 s para 10k linhas; ~22.000× mais lento que XGBoost).
3. **As reversões.** A melhor família é **condicional**: GBDTs/TabPFN lideram binária e
   regressão; **DL reverte à frente em multiclasse** (STab/TabM/RealMLP). GBDTs melhoram
   com o desbalanceamento; STab melhora com o tamanho (ρ=−0.64, p=0.004); modelos de
   atenção quebram no dataset categórico (one-hot explode). E isso é **previsível**: uma
   árvore de meta-features (tamanho, dimensionalidade, fração categórica) recupera quem
   vence.
4. **O framework de decisão (a contribuição).** Síntese em **matriz multicritério**
   (modelo × 10 critérios, pontuada a partir dos dados) + **flowchart prático** —
   inexistente na literatura para esse conjunto 2024–2025.

## Pontos para a reunião
- **Robustez/risco**: TabPFN não é só melhor na média — tem o **maior piso** (raramente
  catastrófico); todos os outros caem a rank 10–11 em algum fold.
- **Honestidade estatística**: N pequeno (10/3/5 datasets). Não afirmo "vencedor único";
  uso Bayesiano + decomposição de variância para sustentar o empate com rigor.
- **Onde mora a AI-2 (proposta)**: o gargalo de **encoding categórico do DL** (one-hot →
  OOM/degradação dos modelos de atenção). Um pipeline de *embeddings aprendidos / target
  encoding* que feche a distância DL–GBDT em dados categóricos é uma contribuição concreta
  e validável, semeada diretamente por este benchmark.

## Próximo compute (escopado honestamente)
- **Curvas de aprendizado / crossover** (RQ2): re-treino por frações para achar o tamanho
  em que DL alcança GBDT.
- **Sensibilidade ao tuning**: re-rodar com defaults para medir o valor do tuning por
  modelo.

## Entregáveis prontos
9 notebooks executados (heatmaps, ranks, Friedman/Nemenyi/CD, **Bayesiano**, Pareto custo,
**latência**, robustez, meta-features, **matriz de decisão + flowchart**), SHAP,
e 3 documentos de análise. Tudo versionado no GitHub.
