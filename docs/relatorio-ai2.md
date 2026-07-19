# Relatório técnico consolidado — AI-2 (Atividade de Orientação Individual)

> Estado em 19/07/2026, após as fases 1 e 2 do programa experimental de
> destilação e a validação completa do framework. Todos os números provêm de
> artefatos versionados (`results/distillation/*`, `results/aggregated/*`,
> `results/latency/*`). Desenho pré-registrado:
> `docs/desenho-experimental-destilacao.md` (+ adendo de execução).

## Sumário executivo

A AI-2 tem dois pilares complementares. O **Pilar B (framework validado)**
está concluído: a validação leave-one-dataset-out estabelece que roteamento
por meta-features não generaliza e que a política "foundation model primeiro,
desvie por restrição" tem regret mediano zero com os 14 modelos. O **Pilar A
(destilação)** completou o piloto decisório e a fase 2: a destilação
**pontual** não retém ganho em escala de pool completo (H1 refutada — com
qualquer teacher); a destilação **distribucional** retém de 13% a 64% da
vantagem de CRPS do teacher (H2 confirmada em 4/5), a latências 3–4 ordens
de magnitude menores. O recorte inédito (regressão distribucional) é
exatamente o que carrega o sinal positivo.

## Pilar B — Framework de decisão validado (concluído)

| Política | Regret médio @11 | Regret mediano @11 | Regret médio @14 | Regret mediano @14 |
|---|---|---|---|---|
| Árvore meta-features (LODO) | 0,228 | 0,145 | 0,051 | 0,000 |
| Sempre-GBDT | 0,249 | 0,168 | 0,305 | 0,291 |
| Sempre-DL | 0,189 | 0,132 | 0,313 | 0,349 |
| **Sempre-FM** | **0,115** | **0,018** | **0,021** | **0,000** |

- Hit-rate da árvore sob LODO: 0,11 vs 0,44 do baseline majoritário — como
  *preditor*, o roteamento por meta-features é refutado (a melhora aparente
  @14 decorre de a classe majoritária ter virado "FM").
- Com a geração 2026, a pergunta "qual família?" se dissolve; o que resta é
  a decisão por **restrições** (latência, nº de classes, interpretabilidade)
  — exatamente a estrutura do flowchart do benchmark, agora com validação
  quantitativa em vez de intuição.
- Artefatos: `scripts/lodo_validation.py`,
  `results/aggregated/lodo_validation{,_14}.csv`.

## Pilar A — Destilação distribucional (fases 1-2 executadas)

### Motivação quantificada (o problema é real)

Latência de inferência medida (adult, µs/linha, mediana de 5 passadas):
XGBoost 0,33 · aluno-quantil XGBoost ~O(µs) · **TabPFN 7.416** ·
**TabFM 43.262** (~131.000× o XGBoost). Os líderes de acurácia do benchmark
são de 4 a 5 ordens de magnitude mais lentos para servir.

### Fase 1 — teacher TabPFN (distribucional), 5 datasets, 3 sementes

- **H1 (ponto, RMSE): REFUTADA.** Alvos soft/misto nunca superam o controle
  hard em pool completo.
- **H2 (distribuição, CRPS): CONFIRMADA onde o teacher tem vantagem.**
  Retenção do gap de CRPS: california **+0,64**, year_prediction **+0,42**,
  superconduct +0,26, diamonds +0,13; wine −0,97 (único dataset onde o
  teacher TabPFN é *pior* que o baseline — falha prevista pelo desenho §6).
- Calibração transferida: no california, PICP80 do aluno destilado 0,60 vs
  0,88 do hard (nominal 0,80) — em year, 0,79 vs 0,71 (destilado mais
  próximo do nominal). Padrão heterogêneo, reportado integralmente.

### Fase 2 — teacher TabFM (pontual; gaps 4× maiores)

Âncoras confirmam gaps bem maiores (california 0,031 vs 0,008 do TabPFN;
wine +0,036 — positivo, ao contrário do TabPFN; year 0,255). Ainda assim:

- A destilação pontual **continua sem ganho confiável**: única célula
  positiva relevante é california/misto (retenção 0,21); nos demais, hard ≥
  soft. Conclusão robusta: **em pool completo, o aluno GBDT não absorve a
  vantagem pontual do foundation model — independente do tamanho do gap.**
- Implicação de desenho: o caminho promissor para o TabFM-teacher é expor a
  distribuição interna (bins) — item exploratório registrado — ou aceitar o
  TabPFN como teacher distribucional canônico.

### Varredura de tamanho de pool (hipótese do contraste smoke-vs-piloto)

Δ = RMSE(hard) − RMSE(soft), teacher TabPFN, 3 sementes:

| pool | california | wine |
|---|---|---|
| 800 | −0,008 | **+0,026** |
| 2.000 | +0,005 | −0,079 |
| 8.000 | −0,007 | −0,081 |
| completo | −0,008 | −0,081 |

O ganho de poucos dados **replica no wine** (positivo em n=800, desaparece a
partir de n=2.000) mas **não no california** (ruído em todos os tamanhos).
Leitura honesta: o moderador não é apenas o tamanho do pool — é
dataset-dependente. Registrado como observação, não como afirmação geral.

### A leitura de Pareto (H3, preliminar)

No california_housing: o aluno-quantil destilado entrega CRPS 0,109 (vs
0,180 do hard e 0,069 do teacher) a latência de GBDT (~µs), enquanto o
teacher cobra 7.416 µs/linha — **retém 64% da vantagem distribucional a
~3 ordens de magnitude menos latência**. Esta é a célula-semente da figura
central do paper; a fronteira completa (destilar × contexto-reduzido ×
ensemble-reduzido) é a próxima etapa.

## Próximas etapas (fase 3)

1. Fronteira de Pareto completa (H3): medir teacher com contexto
   {1k,5k,10k,25k} e ensemble {1,4,8} + latência dos alunos-quantil.
2. Aluno MLP-quantil (a família do aluno importa?).
3. Ablação OOF vs in-sample (réplica do achado Pocket FM em regressão).
4. Extensão de datasets (OpenML-CTR23) para N que suporte inferência.
5. Exploratório: distribuição interna do TabFM (bins) como alvo.
6. Redação do preprint (título de trabalho no desenho; alvo out/2026).

## Enquadramento honesto para orientador/banca

- O resultado positivo (H2) está **exatamente no recorte inédito** — a
  literatura cobriu classificação (Pocket FM); regressão distribucional era
  o gap aberto e é onde o sinal aparece.
- Os resultados negativos (H1; ponto-com-TabFM) são robustos, replicados com
  dois teachers, e delimitam *quando destilar vale a pena* — contribuição em
  si, alinhada ao compromisso pré-registrado de reportar negativos.
