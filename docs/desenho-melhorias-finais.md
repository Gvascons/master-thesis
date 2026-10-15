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

### A-SORT — EXECUTADO (15/10/2026 04h22, `ablation_sort.csv`)

**H-S1 CONFIRMADA.** O sort pós-hoc melhora o CRPS em **20/20**
conjuntos no braço hard (sinal p<10⁻⁵) e **18/20** no soft (p=0,0004);
nunca piora de forma relevante. Efeito mediano pequeno (0,70% hard /
0,32% soft) mas com cauda: até 28,8% (hard) / 11,3% (soft) nos
conjuntos onde o cruzamento é mais danoso. PICP90 fica mais próximo do
nominal com sort em 19/20 (hard) e 14/20 (soft). Fração de linhas com
cruzamento: mediana 96,2% (mín 0, máx 100%) — o cruzamento é a regra,
não a exceção, e o reparo por ordenação é um regularizador consistente
e de custo zero. As conclusões principais NÃO dependem do sort (efeito
mediano <1%); vira parágrafo de apêndice como pré-registrado.

### E4 — EXECUTADO (15/10/2026 04h32, piloto 5 conjuntos, `pilot_conformal.csv`)

**Veredito do gate: destilação e conformalização são COMPLEMENTARES.**
A conformalização split (10% do pool) conserta a cobertura dos DOIS
braços (PICP90: hard 0,77-0,93 → 0,89-0,95; soft 0,71-0,89 → 0,88-0,91)
mas deixa o CRPS essencialmente intacto (movimentos <1% em todos os 10
braços). A vantagem de forma do destilado sobrevive integralmente:
soft+conformal < hard+conformal em CRPS nos 3 conjuntos de vantagem
positiva (california 0,108 vs 0,181; cpu 1,177 vs 1,289; cps88 160,50
vs 160,98), e os 2 reveses pré-declarados (wine, fifa) permanecem
reveses. Leitura: conformal compra CALIBRAÇÃO, destilação compra FORMA
— são ortogonais. Caveat 1 do adendo de 14/10 (exchangeability fraca no
braço soft) permanece declarado; decisão de extensão: NÃO estender aos
20 antes da defesa do resultado principal (o piloto já responde a
pergunta exploratória; extensão exigiria regerar OOF dentro dos 90%).

### E6 — EXECUTADO (15/10/2026 05h18, piloto 5 conjuntos, `pilot_sampling.csv`)

**H-X1 REFUTADA (no balanço).** Nos 3 conjuntos de vantagem positiva, a
variante por amostragem supera o aluno MSE em CRPS apenas em
california (+2,2%); empata em cps88 (−0,02%) e perde em cpu_activity
(−5,6%); nos reveses, perde em ambos (fifa −0,8%, wine −0,7%). Por
(conjunto, semente): 3/15. Custo: 2,8-44× o tempo de treino do MSE
(K=8 amostras por linha). Nuance honesta: a amostragem melhora a
CALIBRAÇÃO onde o MSE subcobre (california PICP80 0,60→0,80; PICP90
0,71→0,87) — mas o E4 mostra que a conformalização compra a mesma
calibração por fração do custo. Decisão pré-registrada: NENHUM claim de
contribuição metodológica (dispensa a checagem de novidade específica);
registrada como variante testada no apêndice, com a leitura "a
calibração se conserta mais barato por conformal do que por amostragem".

### R-TUNE — EXECUTADO (15/10/2026 06h00, `retune_students.csv`)

**H-T1 REFUTADA em 2 de 3 conjuntos — a consequência pré-registrada
(revisão com adendo) aplica-se.** Retenção herdada → retunada:
california **0,644 → 0,185** (o controle hard herdado estava
sub-provisionado: com Optuna próprio, seu CRPS cai de 0,180 para
0,110); cpu_activity 0,500 → 0,396; cps88wages 0,465 → 0,607 (sobe).
A herança de hiperparâmetros NÃO é piso universal: em conjuntos onde o
controle ganha muito com capacidade própria, ela INFLA a retenção
aparente.

O que sobrevive (e importa): (i) o SINAL sobrevive ao re-tuning nos
3/3 — o soft retunado segue melhor que o hard retunado (0,1025 vs
0,1101; 1,161 vs 1,226; 160,52 vs 160,69); (ii) a destilação e o
re-tuning COMPÕEM — o soft também melhora ao ser retunado (california
0,1086 → 0,1025); (iii) leitura notável: o soft HERDADO (0,1086) ≈
hard RETUNADO (0,1101) no california — a destilação entrega de graça,
sem nenhum trial, o que 25 trials de Optuna entregam ao controle;
(iv) a retenção retunada do california (0,185) coincide com a mediana
global reportada (0,19). Caveats: piloto de 3 conjuntos, 25 trials,
validação interna 80/20 do pool, um best_params por braço reutilizado
nas 3 sementes. Consequência nos textos: a limitação "capacidade
herdada" (cap. 7 e AI-2) é promovida de declaração a resultado
quantificado, e a retenção POR CONJUNTO passa a ser lida como
dependente do orçamento de tuning — as manchetes (sinal, mediana,
pareado) não mudam; a narrativa do california ganha o caveat
obrigatório.

### Adendo datado (15/10/2026, pós-auditoria de vazamento) — caveats adicionais

Auditoria adversarial independente de vazamento (leitura linha a linha
dos 7 scripts + verificação empírica dos artefatos: 20/20 parquets OOF
idênticos ao pool recomputado; interseção pool∩teste nula; splits
idênticos entre E3/E4/E6/R-TUNE/A-SORT/extensão): **nenhum vazamento
de dados de teste encontrado**. Quatro ressalvas registradas:

1. **R-TUNE, braço soft (a declarar, espelha o caveat 1 do E4):** o
   split de validação interna do Optuna (20% do pool) participou, via
   contexto do professor, da geração dos alvos OOF em que os candidatos
   soft treinam — contaminação fraca POOL-INTERNA na seleção de HPs,
   sem caminho até o hold-out. Direção conservadora: favoreceria o
   soft, e H-T1 foi refutada mesmo assim (a refutação fica mais
   sólida, não menos).
2. **Estrutura de grupo em conjuntos específicos (desenho, benchmark
   inteiro):** split aleatório i.i.d. em year_prediction (artistas),
   sarcos (trajetórias), kings_county (revendas), fps/video (grades de
   configuração) pode inflar métricas absolutas de TODOS os sistemas
   igualmente e distorcer a magnitude do gap professor−hard
   (denominador da retenção) nesses conjuntos; não afeta a direção das
   comparações pareadas (mesmo split para os dois braços). Vira frase
   de limitação.
3. **Housekeeping:** `oof/tabfm_diamonds.parquet` é órfão da errata
   diamonds/kin8nm (conteúdo = kin8nm); renomear para
   `tabfm_kin8nm.parquet` APÓS a cadeia noturna terminar (não mexer
   em arquivos OOF com job ativo).
4. **Comparabilidade E4:** os CRPS "raw" do E4 vêm de alunos treinados
   em 90% do pool (internamente consistente); não citar lado a lado
   com os números do pipeline principal (100% do pool) como se fossem
   o mesmo modelo.

### Adendo datado (15/10/2026, manhã) — reinvestigação adversarial em 4 frentes

Quatro auditorias independentes (vazamento; justiça das comparações;
recomputação estatística; fidelidade texto-artefato) sobre todo o
material. Resultado global: **nenhuma conclusão central afetada;
reprodutibilidade numérica 100%** (20+ estatísticas recomputadas do
zero, zero divergências além de arredondamento; ~60 números de texto
conferidos). Correções aplicadas em 15/10:

1. **Unilateralidade rotulada em todas as ocorrências** (sinal,
   Wilcoxon, Fisher) nos dois textos; declarado que a manchete
   sobrevive bilateral (sinal p=0,012) e que Wilcoxon (bilateral 0,090)
   e Fisher (bilateral 0,087) não rejeitariam — coerente com seus
   papéis suplementar/pós-hoc.
2. **Multiplicidade declarada**: sob Holm na família confirmatória, as
   conclusões primárias sobrevivem; os instrumentos suplementares não.
3. **Proveniência da população honestamente reescrita** no AI-2: a
   extensão foi completada para o pool CTR23 elegível integral APÓS
   interim não significativo em N=15 (sinal p=0,059); população final =
   suíte elegível inteira, sem margem de seleção.
4. **"Ângulos independentes" → "complementares"** (re-análises das
   mesmas 20 retenções) nos dois textos.
5. **Erro de LaTeX corrigido no AI-2** (linha de comentário engolia a
   limitação do estimador de CRPS no PDF) e **contradição interna
   resolvida** (Limitations dizia que NGBoost/QRF "não foram
   incluídos" enquanto o Friedman de 5 sistemas os usa — rescopado
   para "default forte via E1").
6. **Exceção do adult declarada** (cap. 4 e AI-1): 11 runs DL legados
   com 100 trials mantidos como âncora; excluído dos agregados de
   custo. **Early stopping dos GBDTs declarado**; **exceção do RealMLP**
   (agenda pytabkit) declarada.
7. **Slogan do pareamento suavizado** (legenda do pipeline, AI-2):
   alvo E objetivo mudam juntos no par quantílico.
8. **Estrutura de grupo** (year_prediction/kings_county) declarada como
   limitação; **caveat do braço soft do R-TUNE** declarado nos textos
   (direção conservadora).

**Retificações de registro (números deste doc):** no veredito do
A-GRID, ler "0,85 p.p." onde está "0,84 p.p." (recomputado: 0,849;
truncamento indevido). No veredito do A-SORT, o p do braço soft é
0,0002 unilateral (o 0,0004 registrado era o bilateral; convenção do
trabalho é unilateral). Gates e conclusões inalterados.

**Nota do auditor de texto sobre H-X1 (registrada):** o critério
"iguala ou supera nos conjuntos de vantagem positiva" admite leitura
"na maioria" (2/3 confirmaria); o veredito adotou a leitura estrita e
refutou — movimento contra a candidata a contribuição própria;
consequência pré-registrada aplicada.

**Pendências pós-cadeia:** renomear oof/tabfm_diamonds.parquet →
tabfm_kin8nm.parquet; dedupe de distill.csv/teacher_eval.csv (o job
SEEDS-10 escreveu duplicatas e linhas NaN in place — números
publicados vêm da versão commitada, intactos).

### SEEDS-10 — EXECUTADO (15/10/2026 12h49, grade completa 20/20)

**CRITÉRIO DE ESTABILIDADE CUMPRIDO (com folga de uma ordem de
grandeza).** Re-execução dos alunos quant hard/soft com 10 sementes
(0-9) nos 20 conjuntos: positivos 16/20 → **16/20** (faixa pré-fixada
[14,18]); mediana 0,192 → **0,191** (Δ=−0,001; tolerância ±0,10);
teste de sinal p=0,0059 (inalterado); maior |Δ| de retenção individual
= 0,013 (fps_benchmark); 14 dos 20 conjuntos movem ≤0,003. Nenhum
número publicado muda; a objeção "3 sementes é pouco" está fechada com
medição. Artefatos: distill.csv e extension.csv agora com sementes 0-9
(os números dos textos seguem referidos a 0-2, agora demonstradamente
equivalentes).
