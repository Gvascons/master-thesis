# Desenho experimental — melhorias da fase final (pré-registro)

> Registrado em 05/10/2026, ANTES da execução, conforme contrato de conduta
> (programa §7). Dois experimentos aprovados para fechar limitations
> declaradas nos artigos AI-1/AI-2 antes da dissertação.

## E1 — Baselines distribucionais clássicos (NGBoost, Quantile Random Forest)

**Pergunta.** A limitation declarada na AI-2 ("classical distributional
baselines outside the compared families were not included") esconde um
confundidor? Isto é: um aluno nativo clássico FORTE (NGBoost, QRF) fecha o
gap que atribuímos à destilação?

**Hipóteses (falsificáveis, registradas antes de qualquer resultado):**
- **H-B1:** na maioria dos 20 datasets, o CRPS do aluno destilado é ≤ ao do
  melhor baseline clássico (teste de sinal unilateral sobre os 20; α=0,05).
- **H-B2:** os baselines clássicos não alteram a leitura da fronteira: suas
  latências são de microssegundos (tier dos students), logo competem com o
  aluno destilado em acurácia, não em custo.

**Gates de interpretação (pré-fixados):**
- Se H-B1 confirmar → a regra de decisão ganha a nota "um nativo clássico
  forte não substitui a destilação onde o edge do teacher é grande".
- Se H-B1 refutar (clássico vence na maioria) → a regra de decisão ganha um
  passo obrigatório ("teste NGBoost/QRF antes de destilar") e TODAS as claims
  de vantagem do aluno destilado são rescopadas nos artigos e na dissertação.
  Resultado negativo é reportado com a mesma proeminência (contrato §7.3).

**Protocolo (idêntico ao da AI-2 — comparação controlada):**
- Mesmos 20 datasets, mesmos splits (hold-out 20% seed 42, caps), 3 sementes.
- NGBoost: distribuição Normal (default), early stopping em validação de 10%
  do pool; quantis das 19 τ extraídos da distribuição prevista.
- QRF (`quantile-forest`, RandomForestQuantileRegressor): quantis nativos nas
  mesmas 19 τ; hiperparâmetros default documentados (sem tuning — espelha o
  tratamento dos students, que herdam tuning do benchmark; desvio declarado:
  baselines SEM tuning específico, interpretação limitada a "default forte").
- Métricas: CRPS = 2×média da pinball na grade de 19 τ (MESMO estimador dos
  artigos — comparabilidade interna), PICP80, RMSE (ponto = mediana prevista),
  latência µs/linha (mediana de 5 passadas no hold-out).
- Artefato: `results/distillation/classical_baselines.csv` (uma linha por
  dataset×modelo×semente).

**Custo previsto:** CPU-only; NGBoost no year_prediction (80k) é o pior caso —
se >2h por semente, cap de iterações documentado como adendo datado.

## E2 — Regret multicritério ponderado no LODO (@14)

**Pergunta.** A política "FM primeiro" continua regret-ótima quando a métrica
de decisão pondera acurácia E custo? (Extensão natural declarada no cap. 6 /
limitations da AI-1.)

**Desenho:**
- Score multicritério por modelo×dataset: `S = w·r_acc + (1−w)·r_lat`, onde
  r_acc = rank normalizado [0,1] da métrica primária no dataset e r_lat =
  rank normalizado da latência de inferência (proxy: latência medida no
  adult, única medição padronizada de 14 modelos — LIMITAÇÃO DECLARADA: a
  latência relativa entre modelos é aproximadamente estável entre datasets
  para GBDT/DL; para FMs ela cresce com o contexto, então o proxy é
  CONSERVADOR a favor dos FMs em datasets pequenos e contra em grandes).
- Varredura w ∈ {1.0, 0.9, …, 0.0}; para cada w, regret normalizado das 4
  políticas (árvore-LODO, sempre-GBDT, sempre-DL, sempre-FM) como no estudo
  original (0 = família recomendada contém o melhor S; 1 = só o pior).
- **Hipótese H-W1:** existe w* < 1 a partir do qual "sempre-GBDT" passa a ter
  regret mediano menor que "sempre-FM" — tornando o flowchart (pergunta 1 =
  latência) a forma correta da política, não um refinamento cosmético.
- Artefato: `results/aggregated/lodo_weighted_regret.csv` + figura.

**Interpretação pré-fixada:** qualquer que seja w*, o resultado enriquece o
cap. 6 ("o peso da latência em que a recomendação vira"); não há resultado
"ruim" — o experimento quantifica o trade-off que o framework já codifica
qualitativamente.

### Adendo datado (05/10/2026) — E1, imputação para o QRF

O pacote `quantile-forest` não aceita NaN (3 datasets CTR23 têm valores
faltantes; o NGBoost os trata nativamente via árvores do sklearn >=1.4).
Decisão: imputação por mediana ajustada no pool e aplicada ao teste,
somente no caminho do QRF — política simples, sem vazamento, declarada.
Nenhum resultado já gravado foi alterado; a grade foi retomada do ponto
de falha (runner resumível).

---

## Execução e vereditos (adendo datado, 05/10/2026 — pós-execução)

**E1 — EXECUTADO (120/120 células; `classical_baselines.csv` +
`baselines_analysis.csv`; análise em `scripts/analyze_baselines.py`):**
- **H-B1 CONFIRMADA:** aluno destilado ≤ melhor baseline clássico em
  **17/20** datasets (sinal unilateral p=0,0013; Wilcoxon p=0,0008).
  A vantagem da destilação não é redundante com nativos fortes default.
- **H-B2 CONFIRMADA:** latências dos clássicos em 4-31 µs/linha (tier dos
  students) — competem em acurácia, não em custo.
- Leitura honesta dos 3 reveses (gate pré-fixado): QRF vence o destilado
  em california_housing, physiochemical_protein e wine_quality; no wine
  o QRF bate inclusive o teacher (0,279 vs 0,289) — o passo "teste um
  aluno nativo forte" da regra de decisão ganha instância concreta (QRF),
  e o clássico só bate o CONTROLE hard em 5/20 (o pinball do XGBoost
  segue um controle representativo). Nota adicionada à regra: um nativo
  clássico forte não substitui a destilação onde o edge do teacher é
  grande, mas deve ser tentado antes (custa segundos).

**E2 — EXECUTADO (com as 4 políticas do pré-registro; a árvore-LODO é
avaliada pela recomendação por dataset armazenada em
`lodo_validation_14.csv`, sem refit — a árvore foi treinada para prever o
vencedor de acurácia e é avaliada sob o score ponderado):**
- **H-W1 CONFIRMADA:** sempre-FM só é regret-ótima (mediana 0,000) em
  w ∈ {0,9; 1,0}; em w=0,8 GBDT (0,067) e DL (0,035) já a batem; para
  w ≤ 0,6, sempre-GBDT tem mediana 0,000. Cruzamento entre w=0,9 e 0,8
  (mediana) e entre 0,8 e 0,7 (média).
- Árvore-LODO nunca supera a melhor política fixa em nenhum w (colapsa
  na política FM em w alto — consistente com a explicação do artigo).

---

## E3 — Utilidade de decisão sob custos assimétricos (pré-registro 14/10/2026)

> Registrado em 14/10/2026, ANTES da execução. Motivação: crítica do
> orientador (reunião de 08/10) de que o ganho distribucional "não mostra
> resultado real". E3 traduz o efeito em unidades de decisão do problema.

**Natureza declarada (anti-double-counting).** E3 NÃO é evidência
independente do CRPS: o CRPS é (2×) a média da perda pinball sobre os 19
níveis τ, e o custo ótimo esperado do problema do jornaleiro com razão de
custos cu/(cu+co)=τ é exatamente a pinball em τ. O que E3 acrescenta:
(i) RESOLVE o efeito por τ (o CRPS agrega), respondendo "em que níveis de
serviço o destilado ganha"; (ii) re-expressa o efeito em unidades nativas
do alvo (R$, nota, s); (iii) adiciona o baseline realista do praticante
que só tem um modelo pontual: ponto + resíduo gaussiano global.

**Desenho.** Mesmos 20 datasets, mesmos splits (hold-out 20% seed 42,
caps), 3 sementes, professor TabPFN (rótulos OOF cacheados). Modelos por
semente, hiperparâmetros herdados do benchmark (idêntico ao protocolo da
AI-2):
- `quant_hard`: XGBoost multi-quantil nos rótulos verdadeiros (controle);
- `quant_soft`: XGBoost multi-saída nos 19 quantis OOF do professor
  (o aluno destilado);
- `gauss`: XGBoost pontual nos rótulos verdadeiros + sigma global dos
  resíduos num split de validação interno de 10% do pool (modelo final
  refit no pool completo); quantis = ponto + sigma·z_tau. É o "workaround"
  padrão de quem só tem modelo pontual.
Métrica: custo realizado médio do jornaleiro (pinball) no hold-out, por
τ na grade das 19; decisão primária em τ=0,90 (reserva/cauda — nível de
serviço típico). Quantis sempre ordenados (sort pós-hoc, como no
protocolo).

**Hipóteses (falsificáveis, fixadas antes de executar):**
- **H-D1:** custo realizado do `quant_soft` ≤ `quant_hard` em τ=0,90 na
  maioria dos 20 (sinal unilateral, α=0,05).
- **H-D2:** custo realizado do `quant_soft` ≤ `gauss` em τ=0,90 na
  maioria dos 20 (idem).
- **H-D3 (exploratória, sem teste):** o edge soft−hard cresce com τ
  (a vantagem distribucional vive nas caudas); relatar o perfil por τ.

**Gates de interpretação (pré-fixados):**
- H-D1 confirma → "o ganho em CRPS se materializa no nível de serviço de
  cauda" entra na dissertação/artigo com o perfil por τ.
- H-D1 refuta → reportar com a mesma proeminência: o ganho distribucional
  é detectável no score agregado mas não se traduz em vantagem de decisão
  em τ=0,90 neste desenho; claims de utilidade prática são rescopadas.
- H-D2 refuta (gaussiano cobre) → a regra de decisão ganha o passo
  obrigatório "tente ponto+resíduo paramétrico antes de destilar".

**Artefatos:** `results/distillation/decision_utility.csv` (uma linha por
dataset×modelo×semente×τ) + `scripts/decision_utility.py` (runner
resumível) + `scripts/analyze_decision_utility.py` (testes e perfil).
Teacher no teste NÃO é re-executado (custo GPU); a comparação de E3 é
intra-tier de alunos, declarado.

---

## Rodada de fortalecimento pós-reunião (pré-registro 14/10/2026)

> Registrado ANTES da execução. Origem: revisão adversarial estilo banca
> (14/10) + críticas do orientador (08/10). Quatro itens confirmatórios
> e um piloto exploratório declarado.

### R-COND — retenção condicionada (análise descritiva, já executada)

Pós-hoc DECLARADA (não é teste pré-registrado): retenção vs vantagem
relativa do professor, rel=(CRPS_hard−CRPS_teacher)/CRPS_hard.
Resultado (artefato `retention_conditional.csv`): o condicionante prediz
o SINAL, não a magnitude — metade de edge alto: 10/10 positiva, mediana
0,19; as 4 falhas concentram-se todas na metade baixa (Fisher p=0,043);
Spearman da magnitude n.s. Leitura honesta adotada: o gate da regra
elimina o dano (downside), não garante retenção grande. NOTA DE
CORREÇÃO: a hipótese alternativa "retenção 0,4+ no regime alto" (sugerida
na revisão adversarial a partir do edge ABSOLUTO) foi testada e NÃO se
sustenta — o edge absoluto é dominado pela escala do alvo (Spearman
−0,27 n.s.) e não serve de condicionante.

### SEEDS-10 — robustez das sementes dos alunos (3→10)

Extensão de robustez, sem hipótese nova: re-executar os alunos
(xgb_point/xgb_quant × hard/soft) com sementes 0-9 (rótulos OOF do
professor cacheados; runners resumíveis). Critério pré-fixado: a
manchete 16/20 é considerada estável se a contagem com 10 sementes
ficar em [14, 18] e a mediana de retenção em 0,19 ± 0,10. Fora disso,
os números da dissertação/artigo são ATUALIZADOS para os de 10 sementes
com adendo datado.

### E5 — roteadores adicionais no LODO

Pergunta: o resultado negativo do roteamento (árvore d=2 não supera a
classe majoritária) depende da classe de roteador? Roteadores
adicionais sob o MESMO protocolo LODO (@11 e @14): kNN (k=3) sobre
meta-características padronizadas, regressão logística multinomial
(C=1), árvore d=3. H-R1: nenhum roteador supera o baseline de classe
majoritária em taxa de acerto LODO em nenhuma das duas grades. Se
algum superar → o resultado negativo é rescopado para "árvores rasas"
e o cap. 6 é corrigido com adendo (reportado com a mesma proeminência).
Artefato: `results/aggregated/lodo_routers.csv`.

### A-SORT — ablação do reparo por ordenação

O texto afirma que "transferência + ordenação atua como regularizador de
calibração" sem isolar o componente. Ablação: mesmos alunos quant
hard/soft (3 sementes, 20 conjuntos), métricas com e sem o sort
pós-hoc de quantis. H-S1: o sort melhora (ou não piora) CRPS/PICP em
ambos os alvos; a fração de linhas com cruzamento é reportada por
conjunto. Qualquer que seja o resultado, vira parágrafo do apêndice.
Artefato: `results/distillation/ablation_sort.csv`.

### E4 — PILOTO exploratório: conformalização pós-hoc (declarado)

Pergunta (exploratória; piloto ≠ resultado): a vantagem do aluno
destilado sobrevive quando ambos os alunos recebem calibração conformal
barata (split-conformal por quantil, estilo CQR, com 10% do pool como
calibração)? Se hard+conformal ≈ soft+conformal, a vantagem prática da
destilação é rescopada para "quando não se pode pagar o split de
calibração"; se soft+conformal > hard+conformal, a destilação e a
conformalização são complementares (CRPS mede forma além da cobertura).
Piloto em 5 conjuntos (california_housing, cpu_activity, cps88wages,
wine_quality, fifa — 3 sucessos, 2 falhas), 3 sementes. Decisão de
extensão aos 20 só após o piloto, com adendo. Artefato:
`results/distillation/pilot_conformal.csv`.

### E5 — EXECUTADO (14/10/2026, `lodo_routers.csv`)

**H-R1 CONFIRMADA: 0/8 células superam o baseline de classe
majoritária.** @14: tree_d2 0,722 / tree_d3 0,667 / kNN3 0,722 / logreg
0,556 (baseline 0,778). @11: 0,111 / 0,278 / 0,278 / 0,333 (baseline
0,444). Sanidade: tree_d2 reproduz exatamente os números do estudo
original nas duas grades. O resultado negativo do roteamento não
depende da classe de roteador.

### Adendo datado (14/10/2026, pós-revisão de código) — caveats declarados

Auditoria adversarial independente dos 6 scripts novos: zero achados
críticos, nenhuma via de vazamento do hold-out. Três ressalvas
registradas ANTES de qualquer resultado entrar em texto:

1. **E4 (piloto), braço soft:** os rótulos OOF do professor foram
   gerados por 5-fold sobre o pool COMPLETO; o split de calibração de
   10% do piloto portanto participou (via contexto do professor) dos
   alvos de treino do soft — assimetria fraca de exchangeability que o
   braço hard não tem. Mantida no piloto (declarada); se o E4 for
   estendido aos 20, regerar OOF dentro dos 90%.
2. **R-COND:** o condicionante (edge relativo) e o desfecho (sinal da
   retenção) compartilham o CRPS_hard do MESMO hold-out — ruído em
   CRPS_hard move os dois na mesma direção, e parte do Fisher p=0,043
   pode ser esse acoplamento; além disso o gate usa o edge medido no
   teste (versão oráculo — na prática seria estimado em validação OOF).
   Leitura mantida modesta ("prediz o sinal") com estes dois caveats.
3. **E3, baseline gauss:** sigma = desvio-padrão dos resíduos de
   validação (centrado na média dos resíduos; sem termo de correção de
   viés) — fórmula exata declarada; Wilcoxon reportado como suplementar
   (o pré-registro fixa o teste de sinal); análise bloqueada por guarda
   até a grade completa (20/20).

---

## Segunda leva da rodada de fortalecimento (pré-registro 14/10/2026, 23h30)

> Registrado ANTES da execução. Ataca os itens restantes da revisão
> adversarial (estimador de CRPS; teto de capacidade) e adiciona uma
> variante metodológica própria como piloto declarado.

### A-GRID — estabilidade do estimador de CRPS sob grade densa

Pergunta: as comparações internas (retenções, ordenações) são estáveis
quando o CRPS é estimado com 99 quantis em vez de 19? Desenho: 4
conjuntos (california_housing, cpu_activity, wine_quality e fifa — o
último de cauda pesada, onde o ataque tem mais força), professor TabPFN
re-predito no hold-out na grade τ=0,01..0,99; alunos quant hard/soft
(3 sementes) avaliados na grade densa por interpolação linear monótona
da função quantil (método declarado: os 19 quantis DEFINEM a
distribuição do aluno; a interpolação é a sua extensão canônica).
H-G1: a razão CRPS99/CRPS19 é aproximadamente constante entre sistemas
(variação < 3 p.p. dentro de cada conjunto), preservando ordenações e
retenções. Se violar em cauda pesada → a limitação do estimador é
promovida de nota a parágrafo, com os diagnósticos de fifa rescopados.
Artefato: results/distillation/ablation_densegrid.csv.

### E6 — PILOTO: destilação por amostragem (variante metodológica)

Motivação: a destilação atual ajusta 19 superfícies de regressão por
MSE multi-saída (curve-fitting da função quantil). Variante proposta:
AMOSTRAR K=8 pseudo-rótulos por linha da distribuição do professor
(inversa da função quantil OOF por interpolação; u~U estratificado) e
treinar o aluno quantílico NATIVO (pinball multi-quantil) no pool
aumentado — o aluno faz regressão quantílica genuína sobre a
distribuição do professor, com perda própria, em vez de imitar curvas.
H-X1 (exploratória): o aluno por amostragem iguala ou supera o aluno
MSE em CRPS nos conjuntos de vantagem positiva. Piloto nos mesmos 5
conjuntos do E4; decisão de extensão só após o piloto. Se superar →
candidata a contribuição metodológica (com nova checagem de novidade
específica antes de qualquer claim); se não → registrado como variante
testada no apêndice. Artefato: results/distillation/pilot_sampling.csv.

### R-TUNE — teto de capacidade dos alunos (re-tuning)

Item 3 da revisão adversarial ("herança de hiperparâmetros subestima o
teto?"). Desenho: nos 3 conjuntos de maior vantagem genuína do
professor com retenção positiva (california_housing, cpu_activity,
cps88wages), re-otimizar os hiperparâmetros do aluno quantílico com
Optuna (25 trials, validação interna 80/20 do pool, objetivo = pinball
média na grade de 19), SEPARADAMENTE para hard e soft (cada alvo ganha
sua capacidade ótima); 3 sementes finais com os HPs vencedores.
H-T1: a retenção com re-tuning é >= à retenção com HPs herdados
(a herança é piso, não viés pró-destilação). Se a retenção CAIR com
re-tuning no braço soft → a comparação original é revisada com adendo.
Artefato: results/distillation/retune_students.csv.

### A-GRID — EXECUTADO (15/10/2026 00h, `ablation_densegrid.csv`)

**H-G1 CONFIRMADA.** Razão CRPS99/CRPS19 por sistema (média de
sementes): california 0,9641-0,9725; cpu_activity 0,9648-0,9670;
fifa 0,9642-0,9674; wine 0,9663-0,9709 — spread máximo 0,84 p.p.
dentro de cada conjunto (gate: 3 p.p.), incluindo o alvo de cauda
pesada. Retenções sob a grade densa: california +0,636 (vs +0,644),
cpu +0,498 (vs +0,500), fifa -1,00 (vs -1,29), wine -0,93 (vs -0,97)
— sinais, ordenações e diagnósticos preservados. O viés da grade de
19 é multiplicativo quase constante TAMBÉM em dados reais de cauda
pesada, não só nos cenários gaussianos da validação original.

### E3 — EXECUTADO (15/10/2026 ~01h10; grade completa 180/180;
`decision_utility.csv` + `decision_utility_analysis.csv`)

- **H-D1 CONFIRMADA:** custo realizado do jornaleiro em τ=0,90 do
  aluno destilado <= controle hard em **18/20** (sinal p=0,0002;
  Wilcoxon suplementar p=0,0001).
- **H-D2 CONFIRMADA:** destilado <= ponto+gaussiana em **17/20**
  (sinal p=0,0013; Wilcoxon suplementar p=0,0002).
- **H-D3 (exploratória):** perfil em U — edge mediano do destilado
  vs hard: ~1% no centro (τ=0,45-0,65), subindo a 7,96% em τ=0,90 e
  8,18% em τ=0,95 (e 8,66% em τ=0,05); vs gaussiano: 14,06% em τ=0,90,
  20,58% em τ=0,05. A vantagem distribucional concentra-se nas caudas
  — exatamente onde vivem as decisões de custo assimétrico.
- Nota de leitura (pré-declarada): E3 resolve por τ e re-expressa em
  unidades de decisão o mesmo fenômeno que o CRPS agrega; a contagem
  18/20 > 16/20 reflete que o edge de cauda é mais consistente entre
  conjuntos que o CRPS agregado.
