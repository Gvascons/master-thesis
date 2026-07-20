# Capítulo 2 — Fundamentação Teórica

> Prosa v1 (20/07/2026), expandida do esqueleto com fontes verificadas nos
> memorandos de `docs/`. Referências indicadas por [chave]; o arquivo .bib
> será consolidado na conversão final.

## 2.1 Dados tabulares e aprendizagem supervisionada

Dados tabulares — observações em linhas, atributos heterogêneos em colunas —
são o formato dominante das aplicações reais de aprendizado de máquina, de
crédito e saúde a telemetria industrial. Sua heterogeneidade é constitutiva:
uma mesma tabela mistura variáveis contínuas de escalas distintas, contagens,
categorias nominais de cardinalidade arbitrária e valores ausentes com
semântica própria. Ao contrário de imagens e texto, não há estrutura
espacial ou sequencial compartilhada que uma arquitetura possa explorar por
construção — cada coluna é um eixo semântico independente. Essa ausência de
invariâncias universais explica por que as revoluções arquiteturais de visão
e linguagem não transferiram diretamente para tabelas, e por que o campo
manteve, por uma década, um campeão de outra família.

Este trabalho considera os três tipos canônicos de tarefa supervisionada:
classificação binária, classificação multiclasse e regressão — esta última
tanto em sua leitura pontual (prever um valor) quanto distribucional
(prever uma distribuição preditiva calibrada), distinção central ao
Capítulo 7.

## 2.2 Gradient boosting em árvores de decisão

O gradient boosting [Friedman, 2001] constrói um ensemble aditivo de
árvores rasas, cada uma ajustada ao gradiente do erro residual do conjunto
corrente. As três implementações modernas dominantes — XGBoost [Chen &
Guestrin, 2016], LightGBM [Ke et al., 2017] e CatBoost [Prokhorenkova et
al., 2018] — diferem em regularização, crescimento de árvore
(nível-a-nível vs. folha-a-folha) e, no caso do CatBoost, no tratamento
nativo de variáveis categóricas via *ordered boosting*, que evita o
vazamento de alvo das codificações ingênuas.

O viés indutivo das árvores casa com a natureza tabular: partições
alinhadas aos eixos capturam interações e não-linearidades por atributo;
invariância a transformações monótonas dispensa normalização; e a seleção
gulosa de cortes confere robustez a atributos irrelevantes. Essas
propriedades fundamentam o desempenho consistente da família e sua posição
de padrão de mercado — a hipótese nula contra a qual toda proposta nova em
dados tabulares deve ser medida.

## 2.3 Aprendizado profundo para dados tabulares

A resposta do aprendizado profundo organizou-se em três linhagens. A
primeira moderniza o MLP: RealMLP [Holzmüller et al., 2024] mostra que uma
"receita" cuidadosa (embeddings numéricos, schedules, regularização) leva o
perceptron multicamadas a competir com GBDTs tunados; TabM [Gorishniy et
al., 2025] obtém o efeito de ensemble com custo quase de modelo único via
parametrização eficiente de múltiplas "sub-redes". A segunda linhagem
importa a atenção: TabNet [Arik & Pfister, 2021] combina atenção com
seleção esparsa de atributos; FT-Transformer [Gorishniy et al., 2021]
tokeniza cada atributo e aplica um Transformer padrão; SAINT [Somepalli et
al., 2021] adiciona atenção entre linhas à atenção entre colunas; STab
[Voskou et al., 2024] introduz competição estocástica local (LWTA). A
terceira linhagem, mais recente, são as Kolmogorov-Arnold Networks (KAN)
[Liu et al., 2024]: apoiadas no teorema da representação de
Kolmogorov-Arnold, invertem o desenho do MLP colocando funções univariadas
aprendíveis (B-splines) nas arestas e somas nos nós. Variantes trocam a
base (polinômios de Chebyshev no ChebyKAN; RBFs no FastKAN), e o TabKAN
[Eslamian et al., 2025] organiza essas variantes num framework para dados
tabulares. A recepção empírica das KANs é controversa: sob comparação com
paridade de parâmetros e FLOPs, o MLP vence na maioria das tarefas de
aprendizado de máquina [Yu et al., 2024] — controvérsia que o Capítulo 5
adjudica sob protocolo neutro.

Transversal às linhagens está o gargalo do encoding categórico: one-hot de
alta cardinalidade explode a dimensionalidade (no nosso benchmark,
`amazon_employee` atingiria ~6.900 colunas sem teto), penalizando
especialmente os modelos de atenção — motivação recorrente para embeddings
aprendidos.

## 2.4 Foundation models tabulares e aprendizado em contexto

Os *prior-fitted networks* (PFNs) reformulam a aprendizagem tabular como
inferência em contexto: um Transformer é pré-treinado em milhões de
conjuntos sintéticos amostrados de um prior (tipicamente modelos causais
estruturais) para aproximar a distribuição preditiva posterior; diante de um
conjunto real, o "treinamento" reduz-se a apresentar os exemplos como
contexto e a predição a um passe direto. O TabPFN [Hollmann et al.] e suas
versões v2/v2.5 estabeleceram o paradigma, com um limite nativo de contexto
(50 mil linhas na v2.5) e — ponto central ao Capítulo 7 — uma **cabeça
distribucional** (bar-distribution) que produz quantis arbitrários na
regressão. O TabFM [Google Research, 2026] é uma implementação independente
do paradigma com arquitetura híbrida (atenção de colunas, compressão de
linhas em tokens, Transformer de ICL), checkpoints separados por tarefa e,
até a escrita, sem artigo revisado por pares. O trade-off constitutivo da
família: o conjunto de treino viaja com o modelo a cada predição — custo de
treinamento próximo de zero, custo de inferência ordens de magnitude acima
dos GBDTs (quantificado no Capítulo 5).

## 2.5 Destilação de conhecimento

A destilação [Hinton et al., 2015], herdeira da compressão de modelos
[Bucila et al., 2006], treina um aluno rápido nas saídas de um professor
caro. Para árvores, a linhagem born-again [Breiman & Shang, 1996; Vidal &
Schiffer, 2020] mostra que ensembles podem ser comprimidos com fidelidade.
Em foundation models tabulares, o Pocket FM [Tanna et al., 2026] estabelece
a destilação para *classificação* em 153 conjuntos, com a rotulagem
out-of-fold como salvaguarda metodológica. A extensão à *regressão
distribucional* — destilar a distribuição preditiva, não a média — exige o
ferramental da regressão de quantis: a perda pinball, o CRPS como regra de
pontuação própria para distribuições, e diagnósticos de cobertura
(PICP) e afiação (largura de intervalos). Esse é o instrumental do
Capítulo 7.

## 2.6 Comparação estatística de algoritmos

A comparação de múltiplos algoritmos em múltiplos conjuntos segue o
arcabouço canônico de Demšar [2006]: teste de Friedman sobre ranks por
conjunto (com a correção de Iman-Davenport para N pequeno), post-hoc de
Nemenyi com diagramas de diferença crítica, e Wilcoxon pareado com correção
de múltiplas comparações. Benavoli et al. [2017] acrescentam a camada
bayesiana: o teste signed-rank bayesiano com região de equivalência prática
(ROPE) permite afirmar *equivalência* — o que o arcabouço frequentista não
autoriza — e é o instrumento que sustenta as conclusões de empate do
Capítulo 5. O Capítulo 4 especifica o uso completo, incluindo os limites de
poder com N pequeno (o piso do p-valor do Wilcoxon com N=5) e a análise de
sensibilidade do ROPE.
