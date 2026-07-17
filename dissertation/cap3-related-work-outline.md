# Capítulo 3 — Trabalhos Relacionados (esqueleto v1)

> Estrutura com fontes mapeadas e o posicionamento do nosso trabalho em cada
> eixo. Prosa a redigir em agosto.

## 3.1 Benchmarks de modelos tabulares
- Grinsztajn et al. 2022 ("why do tree-based models still outperform...");
  McElfresh et al. 2024 (quando redes vencem); Gorishniy et al. 2021;
  TabArena (2025, arXiv 2506.16791 — benchmark vivo).
- **Posicionamento:** nosso diferencial não é largura (18 datasets < TabArena),
  e sim (i) profundidade de protocolo por célula (nested CV + tuning
  uniforme + bayesiano/ROPE), (ii) multicritério (latência/custo/robustez
  medidos sob o mesmo protocolo), (iii) inclusão simultânea da geração 2026
  (TabFM) e da família KAN — ambas ausentes dos benchmarks neutros até onde
  verificamos (memos de 13-15/07/2026).

## 3.2 A controvérsia GBDT vs deep learning
- As duas narrativas (2022 vs 2024-25) e como a RQ2 (curvas de aprendizado)
  as reconcilia como afirmações sobre tamanho amostral.

## 3.3 Avaliação de KANs em tabular
- Alegações dos proponentes (TabKAN; TabKANet) vs avaliações céticas (Yu et
  al.; benchmark de Poeta et al.); ausência de KAN nos benchmarks neutros.
- **Posicionamento:** nosso benchmark fornece o teste independente sob
  protocolo uniforme que faltava (com resultado desfavorável às alegações
  fortes — dado parcial @17/07: meio de tabela em binária, fundo em
  multiclasse).

## 3.4 Foundation models tabulares: avaliação e aceleração
- Linha PFN/ICL (TabPFN v2/2.5, TabICL, TabFlex, MotherNet); TabFM.
- Aceleração: TACO (compressão de contexto), caching, redução de ensemble.
- **Posicionamento:** primeira comparação neutra TabPFN×TabFM sob protocolo
  idêntico (até onde verificado em 13/07/2026); e a comparação
  destilar-vs-comprimir-vs-cachear proposta no Capítulo 7.

## 3.5 Destilação de foundation models tabulares
- Pocket FM (2605.18654) e paper-irmão de saúde (2605.18702); TabDistill
  (few-shot); destilação p/ GAMs (interpretabilidade); engine comercial da
  Prior Labs.
- **Posicionamento (tabela de lacunas):** classificação coberta; regressão
  (distribucional) aberta; Pareto multi-estratégia aberto; TabFM-as-teacher
  aberto. Fonte: `docs/memo-novidade-destilacao.md` (verificação datada).

## 3.6 Arcabouços de decisão / meta-learning para seleção de modelo
- Meta-features e recomendação de algoritmo (literatura clássica de
  meta-learning); limitações com N pequeno.
- **Posicionamento:** nosso resultado LODO negativo (roteamento não
  generaliza com N=18) + política constraint-driven validada por regret — 
  honestidade metodológica como contribuição.
