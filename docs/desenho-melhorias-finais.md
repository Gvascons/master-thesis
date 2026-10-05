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
