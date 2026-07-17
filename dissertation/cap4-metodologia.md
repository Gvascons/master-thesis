# Capítulo 4 — Metodologia Experimental

> Rascunho para a dissertação (v1, 17/07/2026). Fonte primária:
> `docs/metodologia-estatistica.md` (especificação auditada) e
> `configs/*.yaml`. Números citados provêm de artefatos versionados no
> repositório. Conversão para o template institucional: ver
> `dissertation/README.md`.

## 4.1 Visão geral e perguntas de pesquisa

Este trabalho organiza-se em torno de três perguntas de pesquisa:

- **RQ1.** Sob um protocolo experimental uniforme e estatisticamente
  rigoroso, as famílias modernas de modelos para dados tabulares (gradient
  boosting, aprendizado profundo e foundation models) diferem de forma
  praticamente relevante em desempenho preditivo?
- **RQ2.** Como a vantagem relativa entre famílias varia com a estrutura do
  problema — tipo de tarefa, tamanho amostral, desbalanceamento e composição
  das features — e, em particular, onde a liderança troca de mãos ao longo do
  eixo de tamanho?
- **RQ3.** Quando o desempenho preditivo se aproxima do empate ou é dominado
  por um modelo operacionalmente caro, que arcabouço de decisão multicritério
  (desempenho, custo de treinamento, latência de inferência, robustez,
  interpretabilidade) pode orientar a escolha na prática — e é possível
  remover, por destilação, o principal obstáculo da política ótima?

A metodologia foi desenhada para que cada resposta seja sustentada por
evidência com proveniência rastreável: todo número reportado na dissertação
origina-se de um artefato versionado (CSV, JSON ou notebook executado) no
repositório do projeto.

## 4.2 Conjuntos de dados

Foram selecionados **18 conjuntos de dados públicos do OpenML**, cobrindo os
três tipos de tarefa supervisionada em dados tabulares: **10 de classificação
binária, 3 de classificação multiclasse e 5 de regressão**. A seleção
priorizou diversidade estrutural: os conjuntos variam de aproximadamente 1
mil a 581 mil amostras (três ordens de grandeza), incluem composições
numéricas, mistas e predominantemente categóricas, e razões de
desbalanceamento de 1:1 a cerca de 16:1 na classificação. Essa diversidade é
condição necessária para a RQ2, que investiga reversões condicionais à
estrutura do problema.

Conjuntos com mais de 100 mil amostras foram subamostrados de forma
estratificada para **100 mil amostras antes de qualquer particionamento**,
prática padrão em benchmarks tabulares para viabilidade computacional
(Gorishniy et al., 2021; McElfresh et al., 2024). A integridade de cada
conjunto é registrada por hash SHA-256 por experimento, permitindo detectar
qualquer alteração silenciosa dos dados entre execuções.

## 4.3 Modelos avaliados

O benchmark compara **14 modelos em três famílias**:

| Família | Modelos | Proveniência da implementação |
|---|---|---|
| Gradient boosting (3) | XGBoost, LightGBM, CatBoost | bibliotecas oficiais |
| Aprendizado profundo (9) | MLP, RealMLP, TabM, TabNet, FT-Transformer, SAINT, STab, KAN, TabKAN | oficiais (RealMLP, TabM), comunidade (TabNet), e **cinco implementações próprias** (MLP, FT-Transformer, SAINT, STab, TabKAN) — as duas últimas e o TabKAN reimplementados dos artigos originais por ausência de pacote maduro; o KAN usa o backbone efficient-kan (MIT) vendorizado com adaptações documentadas |
| Foundation models (2) | TabPFN v2.5, TabFM (Google, 2026) | pacotes oficiais, pesos pré-treinados, operação zero-shot |

A proveniência é reportada explicitamente porque afeta a interpretação: um
desempenho fraco de modelo reimplementado confunde a qualidade do método com
a da implementação. Mitigamos esse risco com verificações de sanidade contra
os resultados publicados (por exemplo, a reimplementação do TabKAN atinge
AUC 0,914-0,915 no conjunto *adult*, compatível com o valor de ~0,90
reportado pelos autores) e documentamos todo desvio de protocolo (por
exemplo, o treinamento do TabKAN com AdamW e mini-lotes, em lugar do L-BFGS
full-batch original, para uniformidade com os demais modelos de aprendizado
profundo).

Os dois foundation models operam em regime **zero-shot** (nenhum ajuste de
hiperparâmetros; o "treinamento" consiste em condicionar o modelo
pré-treinado ao conjunto de treino). Ambos têm limites arquiteturais
declarados: máximo de 10 classes (TabFM) e contexto nativo de 50 mil amostras
(ambos; acima disso aplica-se subamostragem interna com política idêntica
entre os dois, para comparabilidade). O conjunto *helena* (100 classes) é,
portanto, uma exclusão estrutural para ambos — contabilizada e discutida, e
não omitida.

## 4.4 Protocolo experimental

O protocolo segue quatro princípios: separação estrita entre seleção e
medição; estratificação em todos os particionamentos de classificação;
paridade de condições entre modelos; e reprodutibilidade bit a bit onde o
hardware permite.

1. **Hold-out final.** 20% de cada conjunto é separado *antes* de qualquer
   validação cruzada (estratificado em classificação; semente fixa 42). Todo
   resultado final reportado provém exclusivamente desse conjunto, que nenhum
   estágio de seleção de modelo jamais observa.
2. **Validação cruzada aninhada.** O laço externo (5 folds) mede o modelo; o
   laço interno (3 folds) seleciona hiperparâmetros. A unidade de análise
   estatística é o *dataset* (não o fold), evitando pseudo-replicação
   (Seção 4.7).
3. **Otimização de hiperparâmetros.** Optuna com amostrador TPE e
   MedianPruner. Orçamentos por família: **100 trials para gradient boosting
   e 25 para aprendizado profundo** — o custo por trial do segundo é cerca de
   4× maior, e 25 trials de TPE capturam a maior parte do landscape de
   otimização (Akiba et al., 2019). A direção de otimização segue a métrica
   primária de cada tarefa (maximização de AUC; minimização de log-loss e
   RMSE), verificada em auditoria de código.
4. **Treinamento dos modelos profundos.** Máximo de 200 épocas, parada
   antecipada com paciência 20 monitorando a perda de validação, lote 256,
   AdamW; restauração do melhor estado.
5. **Isolamento e robustez.** Cada experimento executa em subprocesso
   isolado; erros de memória disparam redução automática de lote e, no
   limite, poda do trial — uma configuração excessiva não derruba a fila.
6. **Sementes.** Semente global 42; trials recebem sementes derivadas
   determinísticas.

## 4.5 Pré-processamento por família

Cada família recebe a entrada no formato que sua literatura prescreve, com o
ajuste do pré-processador realizado **exclusivamente no treino de cada fold**:

- **Gradient boosting:** codificação ordinal de categóricas e imputação por
  mediana; o CatBoost recebe os índices das colunas categóricas e usa seu
  tratamento nativo.
- **Aprendizado profundo:** padronização das numéricas e one-hot das
  categóricas com teto de 50 categorias por coluna — teto necessário para
  evitar a explosão dimensional (o conjunto *amazon_employee* atingiria
  ~6.900 colunas sem ele, excedendo a memória da GPU nos modelos de atenção).
- **Foundation models:** codificação ordinal; normalização interna própria
  dos modelos.

Duas adaptações específicas do KAN são documentadas como decisões
metodológicas: normalização de camada antes de cada camada KAN e faixa de
grid [-2, 2], necessárias porque B-splines só são definidas dentro do grid e
entradas padronizadas caem majoritariamente fora da faixa original [-1, 1].

## 4.6 Métricas por tipo de tarefa

A métrica primária de cada tarefa é *threshold-free* e sensível à qualidade
probabilística; as secundárias cobrem os ângulos que a primária não vê. A
acurácia é reportada por convenção, mas nunca fundamenta conclusões, por ser
enganosa sob desbalanceamento.

| Tarefa | Primária | Secundárias | Justificativa da primária |
|---|---|---|---|
| Binária (10) | **ROC-AUC** (↑) | KS, log-loss, F1, acurácia | independente de threshold e de prevalência; robusta ao desbalanceamento presente no benchmark |
| Multiclasse (3) | **log-loss** (↓) | F1-macro, F1-weighted, AUC-OvR, acurácia | única *proper scoring rule* que escala a K classes sem agregação arbitrária; sensível às classes raras |
| Regressão (5) | **RMSE** (↓) | MAE, R² | unidade do alvo; penalização quadrática de erros grandes; padrão da literatura |

O KS (Kolmogorov-Smirnov sobre os escores) conecta o benchmark à prática de
crédito e risco; a comparação RMSE×MAE expõe dominância de caudas; o R²
oferece leitura livre de escala entre conjuntos (comparações de RMSE *entre*
conjuntos são evitadas por não serem comensuráveis).

## 4.7 Arcabouço de inferência estatística

A cadeia inferencial segue a prática canônica para comparação de múltiplos
classificadores em múltiplos conjuntos (Demšar, 2006; Benavoli et al.,
2017), acrescida de camada bayesiana e de decomposição de variância. Todo o
arcabouço foi submetido a auditoria de corretude (jul/2026), incluindo
recomputação manual da estatística de Friedman (idêntica: χ²=31,309 nos
dados binários) e cotejo linha a linha da implementação bayesiana com a
formulação de referência.

1. **Friedman (omnibus)** sobre os ranks por conjunto — não assume
   normalidade nem comensurabilidade entre conjuntos. Reportamos o
   qui-quadrado clássico e a **correção de Iman-Davenport** (menos
   conservadora para N pequeno); nos dados binários, ambos rejeitam a
   hipótese nula de igualdade (p=5,2×10⁻⁴ e p=1,1×10⁻⁴).
2. **Post-hoc de Nemenyi** com diagramas de diferença crítica
   (CD = q_α·√(k(k+1)/6N)); a lógica de agrupamento dos diagramas foi
   verificada formalmente (grupos são intervalos contíguos na ordem de
   ranks, com toda dupla interna distando menos que o CD).
3. **Wilcoxon signed-rank par a par com correção de Holm** (método exato;
   magnitudes, não apenas ranks). **Limitação declarada:** com N=5 conjuntos
   de regressão, o menor p-valor bilateral atingível é 0,0625 — rejeição a
   α=0,05 é matematicamente impossível, e após Holm todos os p-valores
   saturam em 1,0 (verificado empiricamente). Conclusões de regressão
   apoiam-se, portanto, em ranks, tamanhos de efeito e análise bayesiana.
4. **Tamanho de efeito:** Cohen's d na variante pareada (d_z), apropriada ao
   desenho de medidas repetidas.
5. **Intervalos de confiança:** bootstrap percentil (10.000 reamostragens).
6. **Análise bayesiana com ROPE** (signed-rank de Benavoli et al., 2017):
   permite afirmar *equivalência prática*, o que o arcabouço frequentista
   não autoriza. Com ROPE de ±0,01 de AUC, 8 dos 10 modelos comparados ao
   líder são praticamente equivalentes. A **análise de sensibilidade do
   ROPE** (0,005/0,01/0,02 → 2/8/10 modelos equivalentes;
   `results/aggregated/rope_sensitivity.csv`) mostra que a conclusão
   qualitativa é robusta na margem de um ponto de AUC — tratada como limiar
   de irrelevância prática na indústria — e que, mesmo no limiar estrito de
   meio ponto, os gradient boosting de topo (XGBoost, CatBoost) permanecem
   equivalentes ao líder. A gradação com o limiar é esperada e reportada
   integralmente.
7. **Decomposição de variância (η²):** parcela da variância de ranks
   explicada pela família; nos dados binários, η²=0,20 — 80% da variância é
   intra-família, fundamentando a tese de que o modelo específico importa
   mais que a família.
8. **Política para células faltantes:** exclusões estruturais (foundation
   models × *helena*) são declaradas; análises que exigem matriz completa
   usam apenas modelos com cobertura total ou excluem o conjunto, com a
   escolha explicitada caso a caso.

## 4.8 Experimentos complementares

- **Curvas de aprendizado (RQ2).** Seis modelos representativos re-treinados
  em fatias crescentes do pool (500 → pool completo; 3 sementes; 5
  conjuntos), com hiperparâmetros fixos do pool completo e o mesmo hold-out
  do benchmark — isolando o efeito do tamanho amostral a capacidade fixa. O
  desenho (re-uso de hiperparâmetros) é declarado como limitação: re-tunar
  por tamanho deslocaria níveis absolutos, não a direção dos cruzamentos.
- **Latência de inferência.** Medida no conjunto *adult* (mediana de 5
  passadas após aquecimento, com sincronização CUDA), em microssegundos por
  linha — o eixo que separa as famílias em até cinco ordens de magnitude.
- **Validação do arcabouço de decisão (LODO).** O recomendador
  meta-features→família é avaliado por *leave-one-dataset-out*, com
  métrica de *regret* normalizado contra políticas fixas — evitando que uma
  árvore descritiva seja lida como preditor validado.
- **Destilação distribucional (RQ3/contribuição).** Desenho pré-registrado
  em documento próprio, com hipóteses falsificáveis, controles de capacidade
  (hiperparâmetros do aluno fixados nos valores tunados do benchmark),
  rotulagem out-of-fold e métricas distribucionais (CRPS, cobertura de
  intervalos). Detalhado no Capítulo 7.

## 4.9 Reprodutibilidade

Semente global fixa; hash SHA-256 dos dados por experimento; configurações
versionadas (`configs/*.yaml`); dependências com versões pinadas (incluindo
o commit exato de pacotes de pesquisa instalados de repositório);
resultados brutos, agregados e notebooks executados versionados em
repositório público. Correções de registro são feitas por commits de
retificação, preservando o histórico.

## 4.10 Ameaças à validade

- **Validade interna:** vazamento de dados é a ameaça central; mitigada por
  hold-out prévio, ajuste de pré-processamento por fold e rotulagem
  out-of-fold na destilação. Implementações próprias podem subestimar
  métodos; mitigada por verificações de sanidade contra valores publicados e
  declaração de desvios.
- **Validade estatística:** N pequeno por tarefa (10/3/5) limita o poder;
  mitigada pela combinação de testes conservadores, tamanhos de efeito,
  análise bayesiana de equivalência e pela declaração explícita dos limites
  (piso do Wilcoxon; multiclasse descritiva).
- **Validade externa:** 18 conjuntos OpenML não esgotam o espaço de
  problemas tabulares; o teto de 100 mil amostras e a GPU única (RTX 5080,
  16 GB) delimitam o regime estudado. Conclusões são enunciadas dentro
  desses limites.
- **Validade de construto:** métricas primárias *threshold-free* e
  multicritério explícito reduzem o risco de otimizar um proxy inadequado;
  a latência é medida em um único conjunto (limitação declarada).
