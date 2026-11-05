# Desenho — E7: análise de complementaridade por instância (XGBoost × foundation models)

**Data do registro: 22/10/2026** (antes de qualquer execução; commit deste arquivo
precede o commit dos resultados). Motivação: perguntas do orientador na reunião
de acompanhamento de 15/10/2026 — *"o que o XGBoost já está aprendendo e o que o
transformer gera de conhecimento adicional? Onde eles concordam e onde
discordam? Quais os casos que o XGBoost acerta e o outro erra, e vice-versa?
Por que melhorou?"*. O benchmark (caps. 5–6) responde **quanto** o foundation
model ganha; este experimento responde **onde** e contribui para o **porquê**.

## 1. O que será feito

Reprodução das predições por instância no hold-out (20%, semente 42 — o mesmo
split do benchmark) para três modelos: `xgboost` (o GBDT de referência da
pergunta do orientador), `tabpfn` e `tabfm` (os dois foundation models), usando
os `best_params` já versionados em `results/raw/*.json` e o mesmo protocolo do
`scripts/train.py` (refit final em 90/10 do pool, mesmo pré-processamento por
família). **Gate de sanidade**: as métricas recomputadas no hold-out devem
reproduzir as salvas em `results/raw` (tolerância |Δ| ≤ 0,001 na métrica
primária; divergência acima disso invalida a análise do conjunto e é reportada).

População: os 17 conjuntos do benchmark em que os três modelos têm célula
válida (18 menos `helena`, exclusão estrutural dos FMs). Classificação: 10
binárias + 2 multiclasse (`covertype`, `jannis`). Regressão: 5 conjuntos
(com a errata de 19/10 declarada: `california_housing` tem alvo binarizado;
`online_news` é o antigo "superconduct").

Quantidades por instância:
- Classificação: rótulo previsto, vetor de probabilidades, acerto/erro;
  margem do XGBoost (binárias: |p − 0,5|); classe verdadeira.
- Regressão: predição pontual e erro absoluto.

Quantidades por conjunto e por par (XGB × TabPFN, XGB × TabFM):
- Taxa de concordância de rótulo previsto; tabela de McNemar
  (b = só XGB acerta, c = só FM acerta) com teste exato binomial bilateral;
- vitórias líquidas do FM por decil de margem do XGBoost e por classe
  (minoritária × majoritária);
- regressão: fração de instâncias em que o FM tem |erro| menor, correlação de
  Spearman entre os erros dos dois modelos, e vitórias líquidas por decil de
  |y − mediana(y)| (centro × cauda do alvo).

## 2. Hipóteses pré-registradas (direcionais, falsificáveis)

- **H-C1 (redundância dominante).** Os modelos compartilham a maior parte do
  sinal: concordância de rótulo previsto XGB×FM **> 85%** na mediana dos
  conjuntos de classificação. *Refutada se* a mediana ficar ≤ 85%.
- **H-C2 (o ganho vive na incerteza).** As vitórias líquidas do FM
  concentram-se nas instâncias de baixa margem do XGBoost: nas binárias, a
  taxa de vitória líquida do FM no quintil de menor margem supera a do quintil
  de maior margem na **maioria** dos conjuntos (teste de sinal unilateral,
  α = 0,05). *Refutada se* não houver maioria significativa.
- **H-C3 (classe minoritária).** Nas binárias, a taxa de vitória líquida do FM
  é maior na classe minoritária que na majoritária na maioria dos conjuntos
  (teste de sinal unilateral). *Refutada se* não.
- **H-C4 (eco das curvas de aprendizado).** A vantagem líquida do FM por
  conjunto (ΔAUC nas binárias; Δlog-loss nas multiclasse; Δ|erro| mediano na
  regressão, sempre FM − XGB no sentido "FM melhor > 0") **decresce** com o
  tamanho do pool de treino: Spearman(n_pool, vantagem) < 0 no conjunto das
  17 células. *Refutada se* ρ ≥ 0.

Interpretação comprometida de antemão: confirmar H-C1 **e** H-C2/H-C3 sustenta
a leitura "o FM não substitui o conhecimento do GBDT; adiciona informação nas
regiões de incerteza/minoria"; refutar H-C2 e H-C3 com H-C4 confirmada
deslocaria a explicação inteiramente para eficiência amostral. Qualquer análise
além das listadas será declarada pós-hoc.

## 3. Custos e execução

TabFM ~43 ms/linha × ~121 mil linhas de hold-out ≈ 1,5 h; TabPFN ≈ 15 min;
XGBoost segundos. GPU única (RTX 5080). Execução resumível por conjunto
(pula parquets existentes). Script: `scripts/complementarity_analysis.py`
(geração) + `scripts/complementarity_report.py` (agregação/figuras).
Artefatos: `results/complementarity/preds_<modelo>_<dataset>.parquet`,
`sanity_check.csv`, `complementarity_pairs.csv`, `complementarity_margens.csv`.

## 4. Relação com o resto do trabalho

- Benchmark (cap. 5): adiciona a camada explicativa "onde o rank médio nasce".
- Framework (cap. 6): se H-C2/H-C3 confirmarem, reforça o critério "FM quando
  a aplicação vive de casos difíceis/minoritários".
- Destilação (cap. 7): já respondeu o análogo em regressão distribucional
  (a vantagem transferível é a forma da incerteza, não a média); este desenho
  fecha o análogo em classificação/ponto sem tocar naqueles artefatos.

## 5. Adendo de execução — 23/10/2026 (pós-execução; nada acima foi alterado)

Execução concluída em 23/10/2026: **52/52 células dentro do gate**
(|Δ| ≤ 10⁻³ na métrica primária; Δ = 0 exato em 42 delas — retificado em
05/11/2026: este adendo e a mensagem do commit `fadfbb0` diziam "49"; a
contagem correta do `sanity_check.csv` é 42, com 9 células do TabPFN
entre 2×10⁻⁷ e 6×10⁻⁶ e uma do TabFM em 4,8×10⁻⁴). Três registros
operacionais, sem efeito nos critérios:

1. O custo real excedeu a estimativa da seção 3 (~20 h de parede vs. ~2 h
   estimadas): o gargalo são os FMs nos conjuntos grandes multiclasse
   (ensemble por classe), não contemplados na conta de ~43 ms/linha.
2. `tabpfn × year_prediction` e `tabfm × year_prediction` falharam por
   OOM de CUDA no processo longo (fragmentação de VRAM) e foram re-rodadas
   em processos dedicados. Para o TabFM foi necessário reduzir o lote de
   predição de 2.000 para 1.000 linhas (predições independentes por linha;
   o gate verificou |Δ| = 4,8×10⁻⁴ contra a métrica salva, dentro da
   tolerância; log em `results/logs/complementarity_rerun_tabfm_yp2.log`).
3. Vereditos (de `complementarity_hipoteses.csv`): **H-C1 CONFIRMADA**
   (concordância mediana 0,950 TabPFN / 0,936 TabFM > 0,85);
   **H-C2, H-C3 e H-C4 REFUTADAS** nos critérios pré-registrados
   (H-C2: 5/10 TabPFN, 7/10 TabFM, sinais n.s.; H-C3: 6/10 e 8/10
   — TabFM marginal, p=0,0547; H-C4: Spearman +0,143 e +0,260, n.s. —
   a vantagem NÃO decresce com o pool nesta população; descritivamente,
   as maiores vantagens do TabFM estão nos conjuntos grandes).
   Os negativos serão reportados com o mesmo estatuto dos positivos.
