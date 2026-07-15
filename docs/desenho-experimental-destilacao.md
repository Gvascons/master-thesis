# Desenho experimental — destilação distribucional de FMs tabulares (AI-2)

> Título de trabalho: *"Destilação de foundation models tabulares além da
> classificação: regressão distribucional e a fronteira de Pareto entre
> destilar, comprimir e cachear."*
> Prior work citado desde o dia 1: Pocket FM (arXiv 2605.18654 — classificação,
> 153 datasets, rotulagem OOF), Prior Labs distillation engine (fechado),
> TabDistill (few-shot), Bucila 2006 / Born-Again Trees (mecânica).
> Ver `docs/memo-novidade-destilacao.md` para o mapa completo de novidade.

## 0. Fatos de API que ancoram o desenho (verificados em 15/07/2026)

| Componente | Capacidade | Implicação |
|---|---|---|
| TabPFN v2.5 `predict(X, output_type="quantiles", quantiles=[...])` | Distribuição preditiva completa (qualquer grade de quantis) | **Teacher distribucional principal** |
| TabFM `TabFMRegressor.predict` | Só ponto (média do ensemble) | Teacher *pontual*; investigar acesso aos bins internos como item exploratório |
| TabFM `TabFMRegressor.predict_oof` | **OOF nativo** (num_folds_for_cv=5) | Rotulagem out-of-fold sem implementação manual |
| XGBoost ≥2.0 (`reg:quantileerror`, multi-quantil) | Aluno de quantis nativo | Aluno GBDT distribucional viável |

## 1. Perguntas de pesquisa e hipóteses (pré-registradas)

- **RQ-D1 (ponto):** quanto do ganho de RMSE do teacher sobre o melhor GBDT
  tunado um aluno destilado retém, a latência de GBDT?
  **H1:** o aluno retém ≥50% do gap (teacher − baseline) em ≥3 dos 5
  datasets, com latência ≤10 µs/linha.
- **RQ-D2 (distribuição):** a destilação de quantis transfere a *qualidade
  distribucional* (calibração/sharpness), e não só a média?
  **H2:** o aluno-quantil destilado supera em CRPS o aluno-quantil treinado
  só nos rótulos duros, e sua cobertura de intervalos (PICP 80/90%) fica a
  ±5 p.p. do nominal.
- **RQ-D3 (Pareto):** no plano acurácia×latência, destilar domina as
  alternativas de aceleração (contexto reduzido / cache) em orçamentos de
  latência baixos?
  **H3:** existe faixa de latência (≲100 µs/linha) onde o aluno destilado é
  Pareto-dominante sobre o teacher com contexto subamostrado.
- Resultados negativos são informativos e publicáveis ("onde e por que a
  destilação de regressão falha") — compromisso de reportar de qualquer modo.

## 2. Fatores experimentais

### Teachers (2)
1. **TabPFN v2.5** (distribucional — quantis nativos). Principal.
2. **TabFM (Google, 2026)** (pontual — média; OOF nativo). Secundário +
   item exploratório: extrair distribuição dos bins internos se viável.

### Alunos (3 famílias)
1. **XGBoost ponto** — treinado na média do teacher (destilação clássica).
2. **XGBoost quantil** — `reg:quantileerror` multi-quantil na grade
   K={0.05,0.1,...,0.95} (19 quantis), treinado nos quantis OOF do teacher;
   pós-processamento de *quantile crossing* por ordenação (isotônica se
   necessário — decisão documentada).
3. **MLP quantil** — cabeça multi-quantil com pinball loss (mesma grade),
   arquitetura do nosso MLP baseline; testa se a família do aluno importa.

### Alvos de treino do aluno (o eixo da destilação)
- **(a) Hard**: rótulos verdadeiros (controle = nosso baseline tunado).
- **(b) Soft**: alvos do teacher via **OOF** (K=5 folds; teacher ajustado em
  K−1, prevê o fold restante — protocolo Pocket FM; para TabFM, o
  `predict_oof` da lib).
- **(c) Misto**: alvo = λ·soft + (1−λ)·hard, λ∈{0.5, 0.8} (ablação).
- **(d) Transfer set aumentado** (ablação, linha Bucila 2006): pool + N
  pontos sintéticos (perturbação MUNGE-style das features) rotulados pelo
  teacher; N ∈ {0.5, 1.0}×|pool|.

### Estratégias de aceleração comparadas (pilar Pareto — RQ-D3)
1. Teacher completo (contexto 50k) — âncora de acurácia.
2. Teacher com contexto subamostrado: max_num_rows ∈ {1k, 5k, 10k, 25k} —
   proxy de "compressão de contexto" (TACO não tem código público verificado;
   se surgir, entra como ponto adicional).
3. Teacher com ensemble reduzido: n_estimators ∈ {1, 4, 8}.
4. Alunos destilados (todas as variantes acima).
5. Baselines clássicos tunados do benchmark (XGBoost/LightGBM/CatBoost).

## 3. Protocolo

- **Datasets:** os 5 de regressão do benchmark (wine_quality 6.5k,
  california_housing 20.6k, superconduct 21k, diamonds 54k, year_prediction
  80k — cobrem 1.5 ordens de grandeza). *Extensão pós-piloto (para o paper):
  +8–10 datasets do OpenML-CTR23/TabArena regressão — barata, pois o custo
  dominante (rotulagem OOF pelo teacher) é pago uma vez por dataset.*
- **Splits:** os MESMOS do benchmark (hold-out 20% seed 42; pool de treino
  idêntico) — comparabilidade total com os 14 modelos já medidos.
- **Tuning dos alunos:** Optuna 25 trials no CV interno de 3 folds (idêntico
  ao protocolo dos modelos DL/GBDT do benchmark), tunando o aluno COM os
  alvos de destilação (o tuning enxerga soft labels, nunca o teste).
- **Sementes:** 3 (0,1,2) para alunos; teacher é determinístico dado o seed.
- **Rotulagem OOF:** custo único por (dataset, teacher): TabPFN ~7 ms/linha
  → minutos; TabFM ~100 ms/linha × pool ≤50k → ~1.4h no pior caso. Cachear
  os alvos OOF em `results/distillation/oof_targets/` (parquet) — pagos uma
  vez, reutilizados por todas as variantes de aluno.

## 4. Métricas

| Eixo | Métricas | Observações |
|---|---|---|
| Ponto | RMSE, MAE | comparáveis ao benchmark existente |
| Distribuição | **CRPS** (aprox. por integral de pinball na grade de quantis), PICP 80/90% + largura média de intervalo (sharpness), curva de calibração | CRPS é a métrica proper para distribuições |
| Custo | latência µs/linha (protocolo do `measure_latency.py`: mediana de 5 passadas, warm-up, CUDA sync), tempo total de pipeline (fit teacher + rotular OOF + tunar/treinar aluno), memória | permite o "custo amortizado vs recorrente" |
| Retenção | (score_aluno − score_baseline)/(score_teacher − score_baseline) | 0 = não reteve nada; 1 = reteve tudo; >1 = superou o teacher (born-again effect) |

## 5. Análise

- Tabela mestre por dataset: todas as células (teacher/aluno/alvo/estratégia).
- **Fronteira de Pareto** acurácia×latência por dataset e agregada (a figura
  central do paper — extensão direta do nosso `pareto_*.png`).
- Retenção média ± dp entre sementes; ICs bootstrap sobre linhas do teste.
- Com N=5 (ou N≈14 pós-extensão): análise descritiva + bayesiano com ROPE
  (nunca Wilcoxon a N=5 — ver `metodologia-estatistica.md` §3.3).
- Ablações: OOF vs in-sample (quantificar o vazamento que o OOF evita —
  réplica do achado Pocket FM em regressão), λ do misto, transfer set.

## 6. Critérios de decisão pós-piloto

Piloto = wine_quality + california_housing (1 teacher TabPFN, alunos XGBoost
ponto/quantil, alvos a/b), ~1 dia de trabalho + horas de GPU:

- **H1 parcial confirmada** (retenção >0 consistente) → executar grade
  completa nos 5 + iniciar extensão de datasets.
- **Retenção ≈ 0** (aluno soft ≈ aluno hard) → investigar: o gap
  teacher−baseline é grande o suficiente nesses datasets? (No benchmark:
  TabFM bate XGBoost em wine 0.575 vs 0.598 e california 0.235 vs 0.264 —
  gap existe.) Se persistir, o paper vira o resultado negativo com análise
  de causa (informativo; baixo dano ao cronograma).
- **Quantile crossing severo / CRPS degenerado** → trocar pós-processamento
  (isotônica) antes de concluir.

## 7. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| GPU ocupada pelo benchmark dos 14 até ~17/07 | Rotulagem OOF agendada pra janela pós-run; desenho e código do aluno são CPU |
| TabFM sem distribuição exposta | TabPFN carrega o pilar distribucional; TabFM entra no pilar ponto/Pareto |
| Pocket FM lançar extensão de regressão antes de nós | Ciclos de ~3 meses na área → preprint até out/2026; o pilar Pareto-3-estratégias segue nosso |
| N=5 datasets no piloto | Extensão OpenML-CTR23 planejada; custo marginal baixo |

## 8. Cronograma dentro da prorrogação

- **jul (agora):** desenho ✓; implementação do harness de destilação
  (CPU-safe) enquanto o benchmark roda.
- **ago:** rotulagem OOF + piloto (2 datasets) → decisão; grade completa.
- **set:** extensão de datasets + ablações + Pareto 3-estratégias.
- **out:** escrita do preprint + integração à dissertação.
