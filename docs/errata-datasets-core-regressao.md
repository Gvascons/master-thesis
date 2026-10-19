# ERRATA — identidade dos datasets do núcleo de regressão (19/10/2026)

> Descoberta em 19/10/2026, durante a auditoria de identidade
> nome↔ID↔shape dos 33 `openml_id` do config — exatamente a checagem
> registrada como pendência na lição 2 de
> `docs/errata-diamonds-kin8nm.md`. Retificação executada no mesmo dia
> (commits de retificação; histórico preservado). Este documento é a
> referência canônica do episódio. Dois achados independentes, ambos
> presentes desde o commit de criação do projeto (`b8d4537`):

## Achado 1 — o dataset rotulado "superconduct" sempre foi o OnlineNewsPopularity

- `openml_id: 44064` é o **OnlineNewsPopularity** do tabular-benchmark
  de Grinsztajn (suíte "regression on categorical and numerical
  features"): 39.644 amostras × 59 features (14 categóricas), alvo =
  **log do número de compartilhamentos** de artigos do Mashable
  (faixa observada [0,69; 13,65] = log(2)..log(~840k)).
- Os metadados antigos do config (`21000 × 81, numerical`) descreviam o
  superconduct pretendido (21.263 × 81 propriedades químicas,
  temperatura crítica em Kelvin) — **nunca o dado carregado**. Mesmo
  mecanismo do episódio diamonds/kin8nm.
- **A validade experimental NÃO é afetada**: OnlineNewsPopularity é uma
  tarefa de regressão legítima e contínua, executada sob protocolo
  íntegro em todos os experimentos (benchmark @14, destilação,
  extensões). Apenas o RÓTULO estava errado.
- **Retificação**: rótulo renomeado para `online_news` em todos os
  artefatos por `scripts/errata_rename_online_news.py` (52 alterações:
  parquet bruto, 14 JSONs de resultados com campo interno, 19 CSVs
  agregados/destilação, 4 parquets OOF, fold_results local); config,
  scripts com listas fixas, tabela mestra, tabelas do Apêndice A e
  figura `dataset_landscape` regenerados; textos corrigidos
  (dissertação, AI-1, AI-2). Interpretações textuais que porventura
  evocavam o domínio físico foram varridas e corrigidas.

## Achado 2 — o california_housing é a versão BINARIZADA (alvo {0,1})

- `openml_id: 44090` é o "california" da suíte **"classification on
  numerical features"** do mesmo tabular-benchmark: as 8 features são as
  da regressão clássica, mas o alvo foi **binarizado pela mediana**
  (valor do imóvel acima/abaixo; verificado empiricamente: 2 valores
  únicos {0,1} no parquet; descrição do OpenML confirma a suíte).
- O benchmark e a cadeia de destilação o trataram como regressão. As
  contas são mecanicamente válidas (RMSE sobre alvo {0,1} = raiz do
  escore de Brier; CRPS/quantis de um alvo Bernoulli são bem definidos
  e idênticos para todos os sistemas comparados), mas o conjunto é
  **degenerado para claims distribucionais**: a "distribuição" alvo é
  uma Bernoulli, e ganhos de forma ali medem calibração de uma
  probabilidade, não regressão distribucional.
- **Decisão (critério de honestidade do contrato §7)**: o conjunto é
  MANTIDO como executado — a população de 20 foi pré-registrada e
  removê-la a posteriori seria curadoria pós-hoc. Em vez disso: (i) a
  natureza binarizada é declarada em todos os textos; (ii) TODOS os
  resultados-manchete ganham análise de sensibilidade com o conjunto
  excluído (`scripts/sensitivity_california_binario.py` →
  `results/distillation/sensitivity_california.csv`).

### Sensibilidade (20 → 19 conjuntos), recomputada em 19/10/2026

| Resultado | Publicado (20) | Sem california (19) | Veredito |
|---|---|---|---|
| Retenção (3 sem.) | 16/20, med 0,192, sinal p=0,006, Wilcoxon p=0,045 | 15/19, med 0,170, sinal p=0,010, **Wilcoxon p=0,067** | sinal sobrevive; **Wilcoxon unilateral perde o corte de 5%** |
| Retenção (10 sem.) | 16/20, med 0,191, Wilcoxon p=0,041 | 15/19, med 0,166, Wilcoxon p=0,062 | idem |
| IC bootstrap mediana | [0,08; 0,44] | [0,08; 0,42] | segue positivo |
| E3 vs hard / vs gauss | 18/20 p=0,0002 / 17/20 p=0,0013 | 17/19 p=0,0004 / 16/19 p=0,0022 | sobrevive com folga |
| Friedman 5 sistemas | χ²=64,6 p=3e-13 | χ²=63,8 p=5e-13, ordenação idêntica | inalterado |
| A-SORT | 20/20 hard, 18/20 soft | 19/19, 17/19 | inalterado |
| R-COND (Fisher) | p=0,043 | p=0,033 | inalterado |
| A-GRID (spread) | 0,85 p.p. | 0,70 p.p. | inalterado |
| E6 amostragem | vence 1/5 | a única vitória ERA o california; **0/4** | refutação FORTALECE |
| E4 conformal | conserta PICP nos 2 braços | padrão mantido nos 4 restantes | inalterado |
| R-TUNE | H-T1 refutada 2/3 (california era o caso forte) | resta 1/2 (cpu_activity refuta, 0,500→0,396; cps88wages não, 0,465→0,607) | lição qualitativa sobrevive; caso-vitrine degradado |
| Benchmark: ranks de regressão | TabFM 1,2 / TabPFN 3,0 | TabFM 1,25 / TabPFN 3,25 (líderes idem; XGBoost 5,0→5,5 troca com RealMLP 4,75) | leitura inalterada |

Leitura honesta: a conclusão qualitativa (sinal positivo dominante,
professor>soft>hard, utilidade de decisão nas caudas) é robusta; o par
de instrumentos frequentista fica com o teste de sinal significativo e
o Wilcoxon marginal (p=0,067) — degradação declarada com a mesma
proeminência do resultado (a versão bilateral do Wilcoxon, 0,090, já
era declarada como não significativa nos textos). Curiosamente, a única
vitória da variante por amostragem (E6) e o caso-vitrine do R-TUNE
estavam exatamente no conjunto degenerado — coerente com a natureza
Bernoulli do alvo (distribuições de 2 pontos premiam mecanismos
diferentes dos de regressão contínua).

## O que NÃO foi feito (e por quê)

- **Não re-executamos** com o california de regressão verdadeiro nem com
  o superconduct verdadeiro: seria mudança pós-hoc da população
  pré-registrada e custaria dias de GPU na véspera da reta final. A
  opção fica registrada como extensão possível (decisão do autor com o
  orientador). O custo estimado: benchmark 14 modelos × 1 dataset
  (tuning completo) ≈ 1-2 dias; cadeia de destilação × 1 dataset ≈
  horas.
- Os documentos HISTÓRICOS datados (pré-registros, memos, drafts
  supersedidos) não foram reescritos — recebem este documento como
  referência (regra: retificação por adendo, nunca reescrita).

## Lições registradas

1. A checagem nome↔ID↔shape da lição 2 da errata diamonds/kin8nm, se
   executada na época, teria pego os dois casos. Executada agora, 33/33
   IDs verificados: os demais 31 conferem (nome, suíte e shape).
2. Suítes curadas (tabular-benchmark) reusam o MESMO nome em variantes
   de classificação e regressão do mesmo dado — a identidade exige
   verificar a SUÍTE de origem, não só o nome.
3. Alvo com cardinalidade ≤ 10 em tarefa de regressão agora é alerta
   automático de auditoria (o wine_quality, notas 3-9, é a exceção
   legítima conhecida).
