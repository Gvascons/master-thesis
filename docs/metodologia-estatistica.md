# Metodologia estatística — especificação e auditoria

> Documento de referência para a dissertação e para a defesa. Escrito após
> auditoria completa do código de avaliação (13-15/07/2026), a pedido do
> orientador: explicitar as métricas por tipo de tarefa, justificar cada
> teste de hipótese, e registrar as verificações de corretude realizadas.
> Código auditado: `src/evaluation/metrics.py`,
> `src/evaluation/statistical_tests.py`, `src/evaluation/enrichments.py`,
> `src/tuning/tuner.py`, `src/data/registry.py`.

## 1. Desenho experimental (resumo do protocolo)

- **Hold-out de 20%** separado *antes* de qualquer validação cruzada
  (estratificado para classificação), com `seed=42` fixo. Todo número
  reportado como "resultado final" vem desse conjunto, que nenhum passo de
  tuning jamais viu.
- **Validação cruzada aninhada**: laço externo de 5 folds (medição), laço
  interno de 3 folds (seleção de hiperparâmetros via Optuna TPE com
  MedianPruner). GBDTs: 100 trials; deep learning: 25 trials (custo por trial
  ~4× maior; TPE captura o grosso do landscape nesse orçamento — Akiba et
  al. 2019). Foundation models (TabPFN, TabFM): zero-shot, sem tuning.
- **Estratificação em todos os splits de classificação**: hold-out, folds
  externos e internos (`StratifiedKFold`), e o split treino/validação de 90/10
  do fit final. Para regressão, KFold com shuffle.
- **Pré-processamento ajustado só no treino de cada fold** (nunca no
  val/teste do fold), por família de modelo.
- **Cap de 100k amostras** por dataset (subamostragem estratificada antes do
  split; prática padrão — Gorishniy et al. 2021, McElfresh et al. 2024).
- Unidade de análise dos testes: **datasets** (N=10 binária, N=3 multiclasse,
  N=5 regressão), não folds — folds do mesmo dataset não são independentes, e
  tratá-los como réplicas inflaria artificialmente o poder (pseudo-replicação).

## 2. Métricas por tipo de tarefa

Regra geral: a **métrica primária** de cada tarefa é *threshold-free* e
sensível à qualidade probabilística/discriminativa do modelo; as secundárias
cobrem os ângulos que a primária não vê. Direção sempre explícita.

### 2.1 Classificação binária (N=10 datasets)

| Métrica | Direção | Papel | Justificativa |
|---|---|---|---|
| **ROC-AUC** | ↑ | **Primária** | Independente de threshold e da prevalência das classes; mede discriminação pura. Robusta ao desbalanceamento presente em vários datasets (até 16:1). É a métrica primária padrão da literatura de benchmark tabular. |
| KS | ↑ | Secundária | Máxima separação entre as CDFs dos escores das duas classes. Padrão da indústria de crédito/risco — conecta o benchmark ao uso aplicado. |
| log-loss | ↓ | Secundária | Qualidade da *calibração* probabilística, que a ROC-AUC ignora (AUC é invariante a transformações monótonas do escore). |
| F1 | ↑ | Secundária | Qualidade no threshold 0.5, visão operacional. |
| Acurácia | ↑ | Secundária | Reportada por convenção; **não** usada para conclusões, pois é enganosa sob desbalanceamento (um classificador trivial acerta 94% num dataset 16:1). |

### 2.2 Classificação multiclasse (N=3 datasets)

| Métrica | Direção | Papel | Justificativa |
|---|---|---|---|
| **log-loss** | ↓ | **Primária** | Única métrica proper-scoring que escala naturalmente para K classes (até 100 no helena) sem esquema de agregação arbitrário; sensível à qualidade das probabilidades em todas as classes, inclusive raras. |
| F1-macro | ↑ | Secundária | Média não ponderada por classe — expõe desempenho nas classes raras. |
| F1-weighted | ↑ | Secundária | Média ponderada — visão "operacional". |
| ROC-AUC OvR (weighted) | ↑ | Secundária | Extensão one-vs-rest da AUC; reportada com a ressalva de que a agregação OvR ponderada tem interpretação menos limpa que a AUC binária. |
| Acurácia | ↑ | Secundária | Convenção; mesmas ressalvas da binária. |

*Por que não acurácia como primária?* Além do desbalanceamento, a acurácia
descarta toda a informação probabilística — dois modelos com a mesma acurácia
podem ter qualidades de incerteza muito diferentes, e é exatamente isso que
distingue as famílias no nosso estudo.

### 2.3 Regressão (N=5 datasets)

| Métrica | Direção | Papel | Justificativa |
|---|---|---|---|
| **RMSE** | ↓ | **Primária** | Na unidade do alvo; penaliza quadraticamente erros grandes (relevante em risco); métrica primária padrão da literatura. |
| MAE | ↓ | Secundária | Robusta a outliers; a comparação RMSE vs MAE revela se o erro é dominado por caudas. |
| R² | ↑ | Secundária | Livre de escala, permite leitura entre datasets (o RMSE não — comparar RMSE *entre* colunas de datasets diferentes não faz sentido e é explicitamente evitado nas tabelas). |

### 2.4 Uso no tuning

O Optuna otimiza a métrica primária da tarefa no CV interno.
**Verificado na auditoria:** a direção é tratada corretamente
(`direction="minimize"` para RMSE; ROC-AUC maximizada; log-loss multiclasse
negada e maximizada — `tuner.py:84-86`, `metrics.py:compute_primary_metric`).

## 3. Testes de hipótese — o pipeline inferencial

A cadeia segue a prática canônica de comparação de múltiplos classificadores
em múltiplos datasets (Demšar, JMLR 2006; Benavoli et al., JMLR 2017), com
uma camada bayesiana adicional:

### 3.1 Friedman (omnibus)
Teste não-paramétrico de medidas repetidas sobre os **ranks por dataset**.
Escolhido porque: (i) não assume normalidade nem comensurabilidade das
métricas entre datasets (AUCs de datasets diferentes não são comparáveis em
valor absoluto — só os ranks são); (ii) é o teste recomendado pela referência
canônica da área (Demšar 2006). Reportamos o qui-quadrado clássico **e a
correção de Iman-Davenport** (estatística F, menos conservadora para N
pequeno — adicionada na auditoria). Nos dados binários: χ²=31.31 (p=5.2e-4) e
F=4.10 (p=1.1e-4) — ambos rejeitam; a existência de *alguma* diferença é
robusta à escolha da estatística.

### 3.2 Post-hoc 1: Nemenyi + diagrama de diferença crítica
Comparações par-a-par baseadas em ranks, com controle de erro familywise
embutido (distribuição studentized range). Serve de base ao **diagrama de
diferença crítica**: CD = q_α·√(k(k+1)/6N) (fórmula de Demšar; verificada na
auditoria contra `studentized_range.ppf`). Modelos ligados pela barra não são
distinguíveis a α=0.05.

### 3.3 Post-hoc 2: Wilcoxon signed-rank par-a-par com correção de Holm
Complementa o Nemenyi usando as **magnitudes** das diferenças (não só ranks
globais). Todas as 55 comparações corrigidas por Holm-Bonferroni (controle
FWER menos conservador que Bonferroni puro). scipy usa o método exato para
N≤25 sem empates — nosso caso.

**Limitação declarada (importante):** com N=5 datasets de regressão, o menor
p-valor bilateral atingível pelo Wilcoxon é 2·(1/2⁵)=0.0625 — **é
matematicamente impossível rejeitar a 0.05**, e após Holm todos os p saturam
em 1.0 (verificado empiricamente na auditoria). Por isso as conclusões de
regressão se apoiam em ranks, efeitos e na análise bayesiana, nunca em
significância do Wilcoxon. O mesmo vale, com menos severidade, para
multiclasse (N=3, análise declaradamente descritiva).

### 3.4 Tamanho de efeito: Cohen's d (variante pareada, d_z)
d_z = média(dif)/dp(dif) sobre os pares por dataset — a variante correta para
desenho pareado (mesmos datasets para todos os modelos). Complementa os
p-valores: significância sem efeito relevante não interessa, e vice-versa.

### 3.5 Intervalos de confiança: bootstrap percentil
10.000 reamostragens da média (seed fixo). Não assume normalidade — coerente
com o restante do arcabouço não-paramétrico.

### 3.6 Análise bayesiana: signed-rank com ROPE
Implementação fiel a Benavoli et al. (2017) / baycomp (verificada na
auditoria: pseudo-observação a priori em 0 com peso 0.6, médias de Walsh,
pesos de Dirichlet — `enrichments.py:bayesian_signed_rank`). Motivação: o
arcabouço frequentista **não pode afirmar equivalência** (não-rejeição ≠
igualdade); a posterior bayesiana com ROPE pode. É o instrumento que sustenta
a manchete do Ato I ("8 de 10 modelos praticamente equivalentes ao líder").

**ROPE = ±0.01 AUC** para binária: 1 ponto percentual de AUC como limiar de
irrelevância prática, escolha alinhada à granularidade típica de decisões de
modelo na indústria. *Recomendação registrada: análise de sensibilidade do
ROPE (0.005/0.02) como robustez adicional — pendente.*

### 3.7 Decomposição de variância (η²)
Parcela da variância dos ranks explicada pela família (GBDT/DL/FM) via
razão de somas de quadrados entre/total. η²=0.20 nos dados binários → 80% da
variância é intra-família, fundamentando a tese de que "o modelo importa mais
que a família".

## 4. Política para células faltantes

- **TabPFN×helena** e **TabFM×helena** não existem (limite arquitetural de
  classes de ambos os foundation models). Política: análises multiclasse que
  exigem matriz completa (Friedman/Nemenyi/CD) usam apenas os modelos com
  cobertura completa OU excluem o helena — a escolha é declarada caso a caso
  na análise, e a seção multiclasse é tratada como descritiva (N=3) de
  qualquer forma.
- 197/198 experimentos originais concluídos; a expansão para 14 modelos segue
  a mesma contabilidade (53 novos experimentos esperados, 52 possíveis +
  1 exclusão estrutural do tabfm×helena).

## 5. Auditoria de corretude (13-15/07/2026) — o que foi verificado

| Item | Método de verificação | Resultado |
|---|---|---|
| Friedman (uso do scipy) | Recomputação manual da estatística via ranks | Idêntico (31.309) ✓ |
| Iman-Davenport | Implementado na auditoria; conferido contra a fórmula de Demšar 2006 | Adicionado; rejeita consistentemente com o χ² ✓ |
| Nemenyi (scikit-posthocs) | Análise da invariância: p-valores dependem só de \|R_i−R_j\|, logo a negação para lower-is-better é inócua (correta por simetria) | ✓ |
| Diagrama CD | Fórmula do CD conferida contra Demšar; lógica de agrupamento provada válida (grupos são intervalos contíguos na ordem de ranks; qualquer par interno dista < CD) | ✓ |
| Wilcoxon+Holm | Confirmado método exato do scipy para N≤25; piso do p-valor para N=5 demonstrado empiricamente | ✓ + limitação documentada |
| Cohen's d | Confirmada a variante pareada (d_z), nomeada explicitamente | ✓ |
| Bayesiano ROPE | Cotejado linha a linha com a formulação de Benavoli 2017/baycomp | Fiel ✓ |
| Direção do tuning | Leitura de `tuner.py` (minimize para regressão) + resultados de regressão sãos | Sem bug ✓ |
| Estratificação | `get_cv_folds` (StratifiedKFold), hold-out e splits 90/10 conferidos | ✓ |
| log-loss multiclasse | Endurecido: `labels=arange(n_cols)` fixa o conjunto de classes; teste sintético com classe ausente passa | Corrigido ✓ |
| KS | Leitura da implementação (CDFs empíricas em thresholds comuns) | Correta ✓ |
| Pseudo-replicação | Confirmado que os testes usam datasets (não folds) como unidade | ✓ |

## 6. Respostas prontas para a defesa (FAQ antecipada)

- **"Por que não ANOVA?"** Métricas entre datasets não são comensuráveis nem
  normais; Friedman opera em ranks por dataset, que é a estrutura correta
  (Demšar 2006).
- **"Por que dois post-hocs?"** Nemenyi é o padrão para o diagrama CD
  (rank-based, todas as comparações); Wilcoxon+Holm usa magnitudes e é mais
  poderoso para pares específicos. Divergências entre eles são discutidas,
  não escondidas.
- **"N pequeno demais?"** Sim, e o texto afirma isso: 10/3/5 datasets dão
  baixo poder. Por isso (i) não afirmamos vencedor único, (ii) usamos
  bayesiano+ROPE para sustentar *equivalência* com rigor, (iii) declaramos o
  piso do Wilcoxon em regressão. O benchmark prioriza profundidade de
  protocolo (nested CV honesto) sobre largura de N.
- **"Por que ROPE de 0.01 AUC?"** Limiar de irrelevância prática alinhado à
  indústria; sensibilidade a 0.005/0.02 é a robustez planejada.
- **"Os folds não são réplicas?"** Não — pseudo-replicação. A unidade é o
  dataset; os folds alimentam apenas a variância intra-dataset (bandas dos
  gráficos e análise de robustez).
- **"Tuning viu o teste?"** Nunca: hold-out separado antes de tudo, seleção
  de HPs exclusivamente no CV interno, pré-processamento ajustado só no
  treino de cada fold.

## 7. Pendências — estado em 18/07/2026

1. ~~Análise de sensibilidade do ROPE~~ **FEITA** (17/07): 0.005/0.01/0.02 →
   2/8/10 de 10 modelos equivalentes ao líder; no limiar estrito, XGBoost
   (P(equiv)=0.65) e CatBoost (0.54) permanecem equivalentes
   (`results/aggregated/rope_sensitivity.csv`).
2. ~~Re-execução @14~~ **FEITA** (18/07): notebooks 01-09 e deck re-executados
   sobre a matriz de 250 células; a análise bayesiana do "empate" ficou
   **explicitamente escopada aos 11 modelos da geração ≤2025** (decisão
   editorial declarada no deck §4/§6.2).
3. ~~CD @14~~ **FEITO**, com a cautela declarada: com k=14 e N=10 o CD é
   largo; a separação do TabFM se sustenta na consistência entre datasets
   (13/18) e no bayesiano, não no Nemenyi.
