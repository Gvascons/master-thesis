> **ERRATA 20/07/2026:** o dataset historicamente rotulado "diamonds" e o kin8nm (OpenML 44980) — ver `docs/errata-diamonds-kin8nm.md`. Rotulos deste documento ja corrigidos; o diamonds real (44979) entrou pela extensao como `diamonds_real`.

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
  superconduct +0,26, kin8nm +0,13; wine −0,97 (único dataset onde o
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

### Fase 3 — a fronteira de Pareto (H3): COMPLETA (19/07)

30 células medidas em 3 datasets (`results/distillation/pareto.csv`; figura
`results/figures/pareto_distill.png/pdf` — paleta validada, marcadores como
codificação secundária). Leitura em dois planos:

- **Plano pontual (RMSE):** o aluno clássico ocupa o regime rápido (2,6-5,3
  µs/linha) — H1 refutada, visível; o TabFM ancora o extremo de acurácia a
  custo brutal (18 ms/linha no california; **394 ms/linha no year**).
- **Plano distribucional (CRPS) — onde H3 se confirma:** abaixo de
  ~250 µs/linha (california) e ~1.300 µs/linha (year), **só os alunos
  existem, e o destilado é o CRPS-ótimo entre eles**: california 0,109@99µs
  (teacher mais barato: 0,094@245µs); year 4,43@169µs contra 4,70@1.319µs do
  teacher mais barato — o teacher precisa de 10.989 µs/linha para superar o
  aluno destilado em CRPS.
- **Achados colaterais:** (i) ensemble de 1 membro do TabPFN ≈ ou melhor que
  8 membros a 1/7 da latência (california: 0,2521 vs 0,2556) — "reduzir
  ensemble" é subestimada; (ii) o último passo de contexto (25k→50k) no year
  só é mensurável com memory-saving forçado e custa 3,3× de latência por
  ΔRMSE de 0,05 — argumento involuntário contra o teacher completo em GPUs
  de 16 GB. (Correção registrada: a alegação anterior de OOM "determinístico"
  no ponto 50k estava errada — commit 64b410a.)

### Fase 4a — ablação OOF vs in-sample (19/07): o achado muda de eixo

Em regressão, alvos in-sample **não** degradam RMSE/CRPS (ficam iguais ou
levemente melhores que OOF nos 4 datasets) — ao contrário do colapso
reportado em classificação. O dano aparece em outro eixo: **a cobertura de
intervalos do aluno cai consistentemente** (PICP80: −6 p.p. california,
−4 kin8nm, −10 wine) — o teacher é sobreconfiante nas linhas do próprio
contexto e transfere essa sobreconfiança ao aluno como intervalos estreitos
demais. **Desconfundido em 20/07 (célula de contexto fixo):** o tripé
`oof / insample_ctx80 / insample` — os dois primeiros com contexto
idêntico de 80%, diferindo só no vazamento — mostra que (i) o vazamento
por si só não degrada RMSE/CRPS (chega a ajudar levemente), (ii) a
**erosão de cobertura é efeito puro do vazamento** (PICP80 do ctx80 abaixo
do OOF nos 4 datasets: −5,1 p.p. california, −4,5 kin8nm, −0,5
superconduct, −4,6 wine), e (iii) o tamanho do contexto contribui quase
nada (insample ≈ insample_ctx80 em tudo). **Leitura final para o paper:**
em regressão, a rotulagem OOF importa *pela calibração do aluno*, não pela
acurácia — refinamento desconfundido e genuíno sobre o Pocket FM.
(`results/distillation/ablation_insample.csv`, 72 linhas, 3 regimes)

### Fase 4b — aluno MLP-quantil (19/07): a família do aluno importa MUITO

- **XGB-quantil:** destilação ajuda de forma consistente (o pinball nativo
  do XGB é um aprendiz distribucional fraco; os alvos do teacher o
  consertam).
- **MLP-quantil:** com rótulos duros já é forte em datasets
  pequenos/numéricos — **iguala o XGB destilado sem teacher nenhum** no
  california (CRPS 0,110 vs 0,109) e o **supera com folga** no kin8nm
  (0,047 vs 0,064). Mas é instável no year (CRPS 9,66, intervalos
  absurdamente largos) — onde **a destilação o resgata** (9,66→4,91).
- **Síntese honesta (condições de valor da destilação):** destilar vale a
  pena quando o objetivo distribucional nativo do aluno é fraco (XGB) ou
  instável (MLP em dados grandes/difíceis); quando existe aluno nativo
  forte (MLP em dados pequenos/numéricos), ele dispensa o teacher. H2
  permanece válida como enunciada (vs controle da mesma família), e esta
  análise cruzada é o contexto que a torna publicável com maturidade.
(`results/distillation/mlp_students.csv`)

### Fase 5 — extensão CTR23 e o quadro final N=15 (20/07)

Extensão executada em 10 datasets verificados da suíte CTR23 (com as
armadilhas de leakage brazilian_houses/wave_energy excluídas na
verificação; episódio da errata diamonds/kin8nm documentado em
`docs/errata-diamonds-kin8nm.md`). **Quadro consolidado da destilação
distribucional em 15 datasets únicos** (`extension.csv` + core):

- Retenção de CRPS **positiva em 11/15** (mediana +0,13; máx +0,64);
  teste de sinal unilateral p=0,059 — sugestivo, não significativo a 5%.
- **As falhas refinam a história:** com N=5, "teacher sem vantagem"
  explicava a única falha; com N=15, três das quatro falhas (fifa −1,29;
  diamonds_real −0,32; health_insurance −0,26) ocorrem COM vantagem do
  teacher — e concentram-se em **alvos de cauda pesada/escala de preço**.
  A pré-condição sozinha não basta; a hipótese viva é a transformação do
  alvo (log) como moderador — célula de ablação log-target registrada
  como pendência.
- Leitura calibrada para o paper: a destilação distribucional ajuda na
  maioria dos casos com retenção mediana de 13%, o efeito é heterogêneo,
  e caracterizamos os modos de falha — enunciado honesto e defensável.

### Fase 6a — ablação log-target (20/07): hipótese refutada, insight melhor

Re-rodamos teacher e alunos-quantil com y'=log1p(y) nos 3 datasets de
falha + 2 controles de preço. **O log NÃO recupera a retenção** (fifa
segue fortemente negativo, −1,67; diamonds_real −0,05; health +0,04) — e
também **encolhe os sucessos** (kings_county +0,21→+0,04). A causa
visível: **as vantagens do teacher praticamente desaparecem no espaço
log** (gaps de CRPS caem à casa de 0,004-0,025) — boa parte da vantagem
distribucional do foundation model em alvos de escala de preço É o
tratamento da escala, que uma transformação barata reproduz.

**Regra de decisão consolidada (o "quando destilar" do paper):** antes de
destilar, tente (i) transformação do alvo e (ii) um aluno nativo forte
(MLP-quantil) — ambos baratos; **destile quando uma vantagem
distribucional real do teacher persistir depois disso** (verificável pelo
OOF barato). Onde ela persiste (california +0,64, abalone +0,60,
cpu_activity +0,50, year +0,42), a destilação entrega a fronteira de
latência. (`results/distillation/ablation_logtarget.csv`)

## Próximas etapas (fase 6 — escrita e refinos)

1. Exploratório: distribuição interna do TabFM (bins) como alvo.
2. Redação do preprint (alvo out/2026) — material empírico FECHADO,
   incluindo a regra de decisão acima como contribuição prática central.

## Enquadramento honesto para orientador/banca

- O resultado positivo (H2) está **exatamente no recorte inédito** — a
  literatura cobriu classificação (Pocket FM); regressão distribucional era
  o gap aberto e é onde o sinal aparece.
- Os resultados negativos (H1; ponto-com-TabFM) são robustos, replicados com
  dois teachers, e delimitam *quando destilar vale a pena* — contribuição em
  si, alinhada ao compromisso pré-registrado de reportar negativos.
