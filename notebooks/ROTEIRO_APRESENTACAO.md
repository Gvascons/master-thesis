# Roteiro da apresentação — Modelos para Dados Tabulares em 2025

> **Como usar isto:** é um roteiro pra você ler/falar, seguindo o notebook
> `00_presentation.ipynb` de cima a baixo. Tom informal, na primeira pessoa.
> Onde tem **[FIGURA: nome.png]** é o ponto de mostrar a imagem correspondente
> (é o mesmo `show(...)` que está no notebook, então basta rolar a célula).
> Os números em **negrito** são os que valem falar com confiança — todos saem
> direto do deck.

---

## Abertura (30 segundos, o gancho)

"A frase que todo mundo repete é: *em dados tabulares, GBDT sempre vence*. XGBoost,
LightGBM, CatBoost — árvore ganha do deep learning, fim de papo.

O problema é que essa frase foi cristalizada **antes** da geração 2024–2025: os
modelos novos de deep learning tabular — TabM, RealMLP, STab — e os *foundation
models* como o TabPFN v2. Então eu peguei essa frase e re-testei com rigor.

E o que eu vou mostrar é o seguinte: **quando a acurácia empata — e ela empata —
a decisão não morre, ela migra pra outros eixos**: custo, latência, robustez,
estrutura da tarefa. Minha contribuição é justamente tornar essa migração
rigorosa e acionável: uma matriz de decisão e um flowchart pro praticante."

**Escala do estudo:** 11 modelos × 18 datasets do OpenML, nested cross-validation
5×3, tuning bayesiano com Optuna, testes estatísticos frequentistas **e**
bayesianos, tudo numa GPU só (RTX 5080).

---

## Parte 1 — Os 11 modelos (e um ponto de honestidade)

"Antes de resultado, um ponto metodológico honesto: **nem todo modelo veio de
graça de uma biblioteca.**"

Fala a proveniência (tem a tabela no notebook):
- **6 de bibliotecas oficiais** — os 3 GBDTs (XGBoost, LightGBM, CatBoost),
  TabPFN, RealMLP e TabM.
- **1 da comunidade** — TabNet (`pytorch-tabnet`).
- **4 implementações próprias em PyTorch** — o MLP baseline, o FT-Transformer
  (arquitetura reimplementada), e **SAINT e STab reescritos a partir dos papers**,
  porque não existe pacote oficial maduro. Isso foi uma parte grande do trabalho
  de engenharia.

"Por que 7 dos 11 são deep learning? Porque é onde mora a diversidade de *viés
indutivo*: atenção (FT-Transformer, SAINT, STab), MLP (MLP, RealMLP, TabM), e
atenção+seleção (TabNet). GBDT é o padrão de mercado, TabPFN é o paradigma novo
zero-shot. Cobrir as três famílias é o que faz disso um mapa do estado da arte,
não um duelo."

**Detalhe que importa (§1.1):** cada família recebe o dado no formato que espera.
O deep learning usa one-hot com **teto de cardinalidade de 50 categorias** — e
esse teto foi *necessário*: sem ele, o `amazon_employee` explodia pra ~6.900
colunas e estourava a memória da GPU nos modelos de atenção. Guarda esse detalhe,
ele volta no final como a semente da AI-2.

---

## Parte 2 — O protocolo (30 segundos, só pra mostrar rigor)

**[FIGURA: o esquema do pipeline — é desenhado na célula, não é imagem externa]**

"O esqueleto é nested cross-validation. O laço externo (5 folds) *mede* o modelo
de forma honesta; o laço interno (3 folds) + Optuna *escolhe* os hiperparâmetros,
sem nunca olhar o teste. Hold-out de 20% separado antes de tudo."

Números pra soltar se perguntarem: seed 42 fixo, **GBDT 100 trials de tuning,
deep learning 25 trials** (o custo por trial do DL é ~4× maior, daí o orçamento
menor). Métrica primária: ROC-AUC na binária, log-loss na multiclasse, RMSE na
regressão.

---

## Parte 3 — Os 18 datasets

**[FIGURA: dataset_landscape.png]**

"O benchmark cobre **3 ordens de grandeza** em tamanho e em dimensionalidade — de
**1 mil a 581 mil** amostras. 10 binárias, 3 multiclasse, 5 regressão. Numéricos,
mistos e categóricos."

**[FIGURA: class_imbalance.png]** — "As razões de desbalanceamento vão de 1:1 a
16:1; isso vira um dos eixos de robustez lá na frente."

---

## ATO I — O Empate

**A ideia central:** "No desempenho bruto, o topo do campo é estatisticamente
**indistinguível**. O teste de Friedman até acusa que existe diferença, mas o
post-hoc não consegue separar o cluster de cima."

**[FIGURA: heatmap_classification.png]** — desempenho modelo × dataset.

**[FIGURA: average_ranks.png]** — "Rank médio por tarefa. Repara: **TabPFN lidera
binária e regressão; o deep learning lidera multiclasse.** Já é uma primeira pista
de que a resposta é condicional."

**[FIGURA: cd_diagram_binary.png]** — "Diagrama de Critical Difference. Todo mundo
ligado pela barra horizontal é estatisticamente empatado. O topo é um bolo só."

Aí eu reforço o empate com **dois instrumentos mais fortes que uma média**:

1. **Teste bayesiano** (signed-rank com ROPE de ±0.01 de AUC): **8 de 10 modelos
   são praticamente equivalentes ao TabPFN** — a massa da distribuição cai dentro
   da zona de equivalência.
2. **Decomposição de variância:** o η² entre famílias é só **0.20**. Ou seja,
   **80% da variância de rank é *intra*-família.** Traduzindo: "GBDT vs deep
   learning" é uma abstração fraca — **o modelo específico importa mais que a
   família**. (Ranks médios por família, de curiosidade: foundation 3.2, GBDT 4.5,
   deep learning 7.0 — mas com essa dispersão interna toda.)

**Fecho do Ato I:** "A manchete não é *quem venceu* — é que *ninguém venceu por
margem perceptível*. O único sinal robusto está no fundo: **o TabNet é
consistentemente o pior**."

---

## ATO II — Os Eixos Ocultos (custo)

**A virada:** "Os modelos que empatam em acurácia são **radicalmente diferentes
em custo**. Tempo de treino varia ~300×; latência de inferência **~26.000×**. E o
ranking de custo é quase o *inverso* da intuição."

**[FIGURA: training_time.png]** — "GBDTs e TabPFN treinam em ~1 a 4 segundos; o
deep learning pesado leva minutos por fit."

**[FIGURA: pareto_binary.png]** — "Fronteira de Pareto custo × desempenho. A
fronteira é **{TabPFN, XGBoost, LightGBM}** — **todo o deep learning é dominado**:
existe sempre alguém mais barato e igual ou melhor."

**[FIGURA: inference_time.png]** — **este é o achado-âncora.** "A **inversão do
TabPFN**: é o mais *barato de treinar* (~1s, zero tuning) e o **2º mais caro de
servir** — umas **22.000× mais lento** que o XGBoost na inferência. Treinar é
grátis, usar em produção é caríssimo."

**Fecho do Ato II:** "Custo é o primeiro eixo onde as famílias separam limpo. E
latência **não** é um corte 'árvore vs rede' — é arquitetural: o tier rápido tem
os 3 GBDTs **e** os deep learning tipo-MLP (RealMLP, MLP, TabM). Só atenção,
in-context (TabPFN) e sampling (STab) é que são lentos."

---

## ATO III — As Reversões (o desempenho é condicional)

**A ideia:** "A melhor família **depende da estrutura do problema**: tarefa,
tamanho, desbalanceamento, tipo de feature."

**[FIGURA: cd_diagram_multiclass.png]** — "Na multiclasse, **STab/TabM/RealMLP (o
deep learning) sobem pro topo**. É a reversão mais marcante do benchmark.
(Honestidade: N=3, então isso é descritivo, não conclusivo.)"

**[FIGURA: imbalance_robustness.png]** — "Sob desbalanceamento: o GBDT *melhora*
relativamente conforme a minoria fica rara; o deep learning piora; TabPFN no meio."

**[FIGURA: feature_type_sensitivity.png]** — "O gargalo do one-hot do deep
learning aparece no extremo categórico."

**[FIGURA: robustness_riskmap.png]** — "Mapa de risco. O **TabPFN tem o melhor
rank médio *e* o maior piso** — ele raramente é catastrófico. Todo o resto tem
pelo menos um fold onde despenca pra rank 10-11."

**Fecho do Ato III:** "GBDT e TabPFN lideram binária e regressão, mas o deep
learning reverte à frente na multiclasse; GBDT fica mais forte sob
desbalanceamento; atenção quebra em categórico. E em risco: TabPFN raramente
falha feio."

---

## §6.1 — Curvas de aprendizado: onde a liderança muda de mãos (RQ2) ⭐ NOVO

> **Esta é a parte nova — a que eu computei depois do benchmark. É o que amarra o
> Ato III e liga direto pra minha próxima contribuição.**

**O setup:** "O Ato III mostrou que o vencedor depende do *tipo* de tarefa. Mas a
tensão original — 'GBDT vence' (Grinsztajn 2022) vs 'o deep learning moderno
alcançou' (TabM, RealMLP 2024) — é, no fundo, uma afirmação sobre **tamanho de
amostra**. Então eu varri isso diretamente: re-treinei cada modelo em fatias
crescentes do pool, de **500 até o pool completo**, com os hiperparâmetros do
Optuna fixos e o **mesmo** teste hold-out. **6 modelos × 5 datasets × 3 sementes.**
Isso isola o efeito do tamanho a capacidade fixa."

**[FIGURA: learning_curves.png]** — "Cada painel é um dataset, eixo-x é o tamanho
do pool em log. Binária em escala linear (AUC, maior é melhor); multiclasse e
regressão em log-y (log-loss e RMSE, menor é melhor — usei log pra não achatar as
curvas, o TabPFN tem valores de ordem de grandeza maior nos tamanhos pequenos)."

**A tabela de crossover** (calculada ao vivo no notebook, pega o melhor de cada
família em cada tamanho):

| dataset | tarefa | vence com pouco dado | vence com muito dado | GBDT reassume em |
|---|---|---|---|---|
| give_me_some_credit | binária | GBDT | GBDT | sempre GBDT |
| higgs | binária | GBDT | **DL** | nunca (DL fica à frente) |
| adult | binária | **DL** | GBDT | n≈4000 |
| jannis | multiclasse | GBDT | **DL** | nunca (DL fica à frente) |
| year_prediction | regressão | GBDT | **DL** | nunca (DL fica à frente) |

"O que ler daqui: em **3 dos 5** datasets (higgs, jannis, year_prediction) o deep
learning ultrapassa os GBDTs conforme o pool cresce e **não devolve a liderança**.
O `adult` mostra o padrão inverso (DL ganha cedo, GBDT reassume por volta de
n≈4000). E só o `give_me_some_credit` fica GBDT o tempo todo."

**E o número mais limpo — o rank no regime de poucos dados (n ≤ 4000):**

| modelo | rank médio |
|---|---|
| **tabpfn** | **1.0** |
| xgboost | 2.8 |
| catboost | 3.6 |
| tabm | 3.8 |
| ft_transformer | 4.4 |
| realmlp | 5.4 |

"**O TabPFN é o melhor em *todos* os cinco datasets no regime de poucos dados** —
rank 1.0 perfeito. É a história 'foundation model mudou o jogo em small-data' de
forma quantificada. E casa com o Ato II: barato de treinar *e* o melhor exatamente
onde o dado é escasso."

**Caveat que eu falo em voz alta:** "Hiperparâmetros fixos do pool completo. Um
re-tune por tamanho mudaria os níveis absolutos, mas a **direção** do crossover é
robusta."

---

## §7 — A ponte: dá pra *prever* o vencedor?

"As reversões não são aleatórias — elas seguem **meta-features** do dataset:
tamanho, dimensionalidade, fração categórica."

**[FIGURA: metafeature_correlations.png]** — correlação entre o rank de cada
modelo e cada meta-feature.

**[FIGURA: winner_decision_tree.png]** — "Uma árvore pequena e interpretável
recupera quem vence: **pequeno + poucas features → TabPFN; grande + poucas → GBDT;
muitas features numéricas → deep learning.** E os splits dessa árvore viram as
ramificações do flowchart. (Descritivo: N=18, sem held-out — ilustra a estrutura,
não é um preditor treinado.)"

---

## ATO IV — O Framework de Decisão (a contribuição da AI-1)

"É aqui que tudo converge numa entrega prática — que **não existe na literatura**
pro conjunto de modelos 2024–2025."

**[FIGURA: decision_matrix.png]** — "Matriz multicritério, pontuada a partir dos
dados. **Nenhum modelo é 3 estrelas em tudo.** O TabPFN domina performance,
tuning e robustez, mas é fraco em latência; os GBDTs são os all-rounders."

**[FIGURA: decision_flowchart.png]** — "E o flowchart do praticante: responde 3
perguntas — precisa de latência baixa? qual o tamanho/dimensão? tem muito
categórico? — e sai um shortlist defensável de modelos pra testar."

---

## §9 — Interpretabilidade

**[FIGURA: feature_importance_comparison.png]** — "Os 3 GBDTs concordam bastante
sobre o que importa (no adult)."

**[FIGURA: shap_beeswarm_adult.png]** — "SHAP no XGBoost: atribuição por amostra,
direção e magnitude de cada feature. GBDT é intrinsecamente interpretável — SHAP
exato via TreeExplainer."

**[FIGURA: prediction_agreement.png]** — "Concordância de predições (kappa de
Cohen) entre famílias: mostra o quanto elas capturam padrões diferentes."

"Resumo do eixo: GBDT é interpretável de nascença; atenção é parcialmente
interpretável; MLP e foundation são caixa-preta, exigem método post-hoc."

---

## Fecho — Honestidade e onde mora a AI-2

**Os caveats que eu carrego no texto (falar isso mostra maturidade):**
- **Baixo poder estatístico** (10/3/5 datasets) — por isso eu não afirmo "vencedor
  único"; uso bayesiano + decomposição de variância pra sustentar o empate.
- **TabPFN em dados grandes** roda em subamostra de 50K — a história real é a
  competitividade *zero-shot*.
- **Multiclasse é sub-amostrada** (N=3) e o TabPFN nem completa o helena (100
  classes > limite) → descritivo.
- O bin severo de desbalanceamento está **confundido** com cardinalidade
  (amazon_employee).

**O que já foi feito (semente de AI-2):**
- **As curvas de aprendizado / crossover (RQ2)** — a §6.1 que acabei de mostrar.

**O que vem a seguir:**
- **Sensibilidade ao tuning** — re-rodar com defaults pra medir o *valor do tuning*
  por modelo.

**E onde mora a contribuição da AI-2** (o fecho forte): "O gargalo mais concreto e
acionável que esse benchmark revelou é o **encoding categórico do deep learning** —
aquela explosão de one-hot que estourava a GPU no amazon_employee. Um pipeline de
**embeddings aprendidos / target encoding** que feche a distância deep learning ↔
GBDT em dados categóricos é uma contribuição validável. E — o melhor — agora é
**mensurável na própria curva da §6.1**: sucesso vai ser **mover o envelope do
deep learning pra esquerda**, ou seja, alcançar os GBDTs com *menos* dados. Não é
um vago 'fechar a distância', é um alvo concreto num gráfico."

**Reprodutibilidade (se perguntarem):** seed 42 fixo, hash SHA-256 de cada dataset
gravado por experimento, configs versionadas, 197/198 resultados + notebooks
executados no GitHub.

---

## Colinha de números (pra não travar)

| Número | Valor | Onde |
|---|---|---|
| Modelos × datasets | 11 × 18 | abertura |
| Split de tarefas | 10 binárias / 3 multi / 5 regressão | Parte 3 |
| Faixa de tamanho | 1K – 581K amostras | Parte 3 |
| Trials de tuning | GBDT 100 / DL 25 | Parte 2 |
| Equivalentes ao TabPFN (bayesiano) | 8 de 10 | Ato I |
| Variância de rank intra-família | 80% (η²=0.20) | Ato I |
| Pior modelo consistente | TabNet | Ato I |
| Variação de tempo de treino | ~300× | Ato II |
| Variação de latência | ~26.000× | Ato II |
| Fronteira de Pareto (binária) | TabPFN, XGBoost, LightGBM | Ato II |
| TabPFN: quão mais lento que XGBoost | ~22.000× na inferência | Ato II |
| Crossover DL→GBDT: DL vence grande em | 3 de 5 datasets | §6.1 |
| Rank do TabPFN em n≤4000 | 1.0 (melhor em todos) | §6.1 |
| adult: GBDT reassume em | n≈4000 | §6.1 |

---

## Mapa rápido: figura → arquivo

Todas em `results/figures/` (`.png` embutida no notebook, `.pdf` pra tese):

- `dataset_landscape.png`, `class_imbalance.png` — os datasets
- `heatmap_classification.png`, `average_ranks.png`, `cd_diagram_binary.png` — Ato I
- `training_time.png`, `pareto_binary.png`, `inference_time.png` — Ato II
- `cd_diagram_multiclass.png`, `imbalance_robustness.png`,
  `feature_type_sensitivity.png`, `robustness_riskmap.png` — Ato III
- `learning_curves.png` — §6.1 (RQ2, a parte nova)
- `metafeature_correlations.png`, `winner_decision_tree.png` — a ponte §7
- `decision_matrix.png`, `decision_flowchart.png` — Ato IV (a contribuição)
- `feature_importance_comparison.png`, `shap_beeswarm_adult.png`,
  `prediction_agreement.png` — interpretabilidade §9

*Detalhes técnicos e o código que gera cada número estão no `00_presentation.ipynb`
(seguindo as mesmas seções). As curvas de aprendizado têm o notebook dedicado
`10_learning_curves.ipynb`.*
