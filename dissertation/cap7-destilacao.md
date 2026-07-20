> **ERRATA 20/07/2026:** o dataset historicamente rotulado "diamonds" e o kin8nm (OpenML 44980) — ver `docs/errata-diamonds-kin8nm.md`. Rotulos deste documento ja corrigidos; o diamonds real (44979) entrou pela extensao como `diamonds_real`.

# Capítulo 7 — Destilação Distribucional de Foundation Models

> Rascunho v1 (19/07/2026). Fontes: `docs/desenho-experimental-destilacao.md`
> (pré-registro + adendo), `docs/relatorio-ai2.md`,
> `results/distillation/*`. A fronteira de Pareto (§7.6) será completada com
> `results/distillation/pareto.csv` (medições em andamento).

## 7.1 Motivação: o preço do novo líder

O Capítulo 5 estabeleceu que a geração 2026 de foundation models quebra o
empate histórico em acurácia; o Capítulo 6, que a política validada de
escolha é "foundation model primeiro, desviando por restrição". A restrição
dominante é a latência de inferência: medimos 7.416 µs/linha para o TabPFN e
43.262 µs/linha para o TabFM no conjunto adult — respectivamente ~22.000× e
~131.000× o XGBoost (0,33 µs/linha). Modelos de in-context learning carregam
o conjunto de treino como contexto em cada predição; o custo é constitutivo,
não acidental. Este capítulo investiga a remoção desse obstáculo por
**destilação de conhecimento**: treinar um aluno rápido (XGBoost) nas saídas
do foundation model.

## 7.2 Posicionamento e recorte

A destilação de foundation models tabulares para *classificação* foi
estabelecida por Pocket FM (Tanna et al., 2026; 153 datasets, rotulagem
out-of-fold, alunos GBDT/MLP) e existe como produto fechado (Prior Labs). O
recorte inédito deste trabalho, verificado por checagem de novidade datada
(15/07/2026): (i) **regressão distribucional** — TabPFN v2.5 produz
distribuições preditivas completas, e destilá-las (não apenas a média) em um
aluno de quantis não tem trabalho público; (ii) a **fronteira de Pareto**
entre três estratégias de aceleração (destilar × comprimir contexto ×
reduzir ensemble); (iii) TabFM como teacher. Adotamos a rotulagem
out-of-fold de Pocket FM como base metodológica — o teacher que pontua o
próprio conjunto de treino produz alvos colapsados — e não reivindicamos
prioridade sobre a formulação geral.

## 7.3 Desenho experimental (resumo; pré-registro no Apêndice)

Hipóteses pré-registradas: **H1** (ponto) — o aluno destilado retém ≥50% do
gap teacher−baseline em ≥3/5 datasets a ≤10 µs/linha; **H2** (distribuição)
— o aluno-quantil destilado supera o controle de rótulos duros em CRPS, com
cobertura de intervalos a ±5 p.p. do nominal; **H3** (Pareto) — existe faixa
de latência onde destilar domina as alternativas. Compromisso explícito de
reportar resultados negativos.

Protocolo: mesmos splits do benchmark (hold-out seed 42); alunos XGBoost com
os hiperparâmetros tunados do benchmark (comparação controlada — só o alvo
muda); alvos duro/soft-OOF/misto; grade de 19 quantis; 3 sementes; métricas
RMSE/MAE, CRPS (integral de pinball), PICP80/90 e larguras; retenção
normalizada do gap. Teachers: TabPFN v2.5 (distribucional) e TabFM
(pontual), ambos com contexto limitado a 50 mil linhas — a política do
benchmark.

## 7.4 Resultados — fase 1 (teacher TabPFN)

**H1 é refutada em escala de pool completo.** Em nenhum dos cinco conjuntos
o aluno pontual destilado supera o controle com rótulos verdadeiros.

**H2 é confirmada onde o teacher tem vantagem.** Retenção do gap de CRPS:
california_housing **+0,64**, year_prediction **+0,42**, superconduct
+0,26, kin8nm +0,13; wine_quality −0,97 — o único conjunto em que o
próprio teacher é inferior ao baseline, condição de falha antecipada pelo
pré-registro. O aluno-quantil destilado herda parte substancial da qualidade
distribucional do teacher operando na casa de microssegundos por linha.

## 7.5 Resultados — fase 2 (teacher TabFM) e o efeito do tamanho

Com o TabFM, os gaps de âncora são maiores (california 0,031 vs 0,008;
wine passa a +0,036; year 0,255). Ainda assim, a destilação pontual não
apresenta ganho confiável (melhor célula: california/misto, retenção 0,21).
A conclusão negativa de H1 é, portanto, **independente do teacher**: em pool
completo, o aluno GBDT não absorve a vantagem pontual do foundation model.

A varredura de tamanho de pool (caps 800/2.000/8.000) testou a hipótese,
sugerida pelo contraste smoke-vs-piloto, de que o ganho pontual é um
fenômeno de poucos dados: ele replica em wine_quality (Δ=+0,026 em n=800,
desaparecendo a partir de n=2.000) mas não em california_housing (ruído em
todas as escalas). Registramos o efeito como observação com moderador
dataset-dependente, não como afirmação geral.

## 7.6 A fronteira de Pareto (H3)

Medimos, no hold-out do benchmark, acurácia (RMSE; CRPS para os sistemas
distribucionais) e latência de inferência (µs/linha) de cada estratégia de
aceleração nos três conjuntos com gap positivo: teacher com contexto
comprimido {1k, 5k, 10k, 25k, cheio}, teacher com ensemble reduzido {1, 4},
alunos destilados e a âncora TabFM (Figura 7.X,
`results/figures/pareto_distill.png`; dados em
`results/distillation/pareto.csv`).

**Leitura em dois planos, correspondendo a H1 e H2:**

- **Plano pontual (RMSE):** o aluno clássico de rótulos duros já ocupa o
  regime de baixa latência (2,6 µs/linha no california com RMSE 0,260) — a
  refutação de H1 tornada visível: não há o que a destilação pontual
  adicionar ali. As curvas do teacher compram acurácia com latência
  (california: RMSE 0,299→0,256 entre 245 e 4.045 µs/linha), e o TabFM
  ancora o extremo de acurácia a 18.159 µs/linha.
- **Plano distribucional (CRPS):** o aluno-quantil destilado é o único
  habitante da fronteira abaixo de ~250 µs/linha — california: CRPS 0,109 a
  99 µs contra 0,181 do controle a 50 µs e 0,094 do teacher mais barato a
  245 µs. Confirma H3 no recorte que importa: **para servir distribuições
  em orçamentos estritos de latência, destilar é a única opção no menu** —
  e retém a maior parte da vantagem do teacher (§7.4).

**Achado colateral — o ensemble do TabPFN:** reduzir de 8 para 1 membro
*melhorou* o RMSE no california (0,252 vs 0,256) a 1/7 da latência (589 vs
4.045 µs/linha) — a estratégia "reduzir ensemble" é mais forte do que o
default da biblioteca sugere, e entra na fronteira em faixas intermediárias.

**Nota de hardware:** o ponto do year_prediction a contexto de 50k só é
mensurável na RTX 5080 (16 GB) com o modo de economia de memória do TabPFN
**forçado** (a heurística "auto" subestima o pico e estoura a VRAM); o custo
desse modo aparece na própria medição — 148,6 ms/linha, ~3,3× a latência do
ponto de 25k (45,1 ms/linha) para um ganho de RMSE de apenas 8,661→8,610.
Em GPUs de 16 GB, o último passo de contexto compra pouquíssima acurácia por
um multiplicador severo de latência — um argumento adicional, e involuntário,
a favor das alternativas ao teacher completo.

## 7.7 Discussão

- **O que a destilação transfere:** não a média (H1), mas a *forma* da
  distribuição preditiva (H2) — consistente com a visão de que o valor
  agregado do foundation model em regressão está na incerteza calibrada,
  que o aluno de rótulos duros não consegue aprender do dado bruto na mesma
  quantidade.
- **Quando vale a pena:** teacher com vantagem real sobre o baseline
  (pré-condição verificável via OOF barato) e demanda por predição
  distribucional em produção de baixa latência.
- **Destilação como suavizador de calibração:** as curvas de
  confiabilidade (Fig. 7.Y) mostram o aluno destilado mais próximo da
  diagonal que o controle — e, no california_housing, que o próprio
  teacher, cuja calibração por nível de quantil é imperfeita (forma-S). A
  combinação transferência-de-curvas + ordenação atua como regularizador
  de calibração, não mera cópia.
- **Limitações:** 5 conjuntos (extensão OpenML-CTR23 planejada); aluno único
  (XGBoost; MLP-quantil planejado); TabFM sem distribuição exposta; CRPS
  aproximado pela grade de 19 quantis.

## 7.8 Síntese

A contribuição delimita com precisão o espaço em que destilar foundation
models tabulares para regressão funciona: o eixo distribucional — 
exatamente o recorte sem trabalho público anterior — com retenção de até
64% da vantagem de CRPS a três ordens de magnitude menos latência, e dois
resultados negativos robustos (ponto em pool completo; independência do
teacher) que orientam a prática e a pesquisa subsequente.
