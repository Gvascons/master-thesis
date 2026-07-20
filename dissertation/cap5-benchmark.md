# Capítulo 5 — O Benchmark Multicritério: Resultados

> Prosa v1 (20/07/2026). Números de `results/aggregated/test_results.csv`
> (250/252 células), `notebooks/TABELA_RESULTADOS.md`,
> `results/latency/latency_adult.csv` e notebooks 01-10. Figuras referidas
> pelos arquivos em `results/figures/`.

## 5.1 Cobertura e organização

O benchmark completa 250 das 252 células possíveis (14 modelos × 18
conjuntos); as duas ausências são exclusões estruturais documentadas — o
conjunto helena (100 classes) excede o limite arquitetural dos dois
foundation models. A exposição segue quatro atos: o empate da geração
até 2025 (§5.2), os eixos de custo (§5.3), as reversões condicionais e as
curvas de aprendizado (§5.4), e a quebra do empate pela geração 2026
(§5.5). O episódio de retificação de rótulo (o conjunto historicamente
chamado "diamonds" é o kin8nm; `docs/errata-diamonds-kin8nm.md`) precede
todos os números aqui reportados.

## 5.2 Ato I — o empate da geração até 2025

Restrita aos 11 modelos disponíveis até 2025, a classificação binária
exibe um topo estatisticamente indistinguível. O omnibus rejeita a
igualdade global (Friedman χ²=31,3, p=5,2×10⁻⁴; Iman-Davenport F=4,1,
p=1,1×10⁻⁴), mas o post-hoc não separa o pelotão superior. Dois
instrumentos mais finos sustentam a leitura de empate: o teste bayesiano
signed-rank com ROPE de ±0,01 de AUC declara 8 dos 10 modelos praticamente
equivalentes ao líder de rank (TabPFN), e a análise de sensibilidade do
limiar (0,005/0,01/0,02 → 2/8/10 equivalentes) mostra que a conclusão vale
na margem de um ponto de AUC — com XGBoost (P(equiv)=0,65) e CatBoost
(0,54) equivalentes mesmo no limiar estrito. A decomposição de variância
fecha o argumento estrutural: η²=0,20 — oitenta por cento da variância de
rank é intra-família, de modo que "GBDT vs. DL" é uma abstração fraca; o
modelo específico importa mais que a família. O único sinal robusto do
desempenho bruto está no fundo: TabNet é consistentemente o pior da
geração.

## 5.3 Ato II — os eixos de custo

Os modelos que empatam em acurácia diferem radicalmente em custo. O tempo
de treino mediano varia ~300× (LightGBM/XGBoost ~0,7 s; SAINT/STab ~200 s);
a latência de inferência varia cinco ordens de magnitude. Medida no
conjunto adult (mediana de 5 passadas): XGBoost 0,33 µs/linha; o tier
rápido inclui os três GBDTs e os DL tipo-MLP (CatBoost 1,4; RealMLP 1,5;
TabKAN 2,3; MLP 2,4; KAN 3,1; TabM 3,5); a atenção custa dezenas de µs
(TabNet 17; FT-Transformer 29; SAINT 45); e o extremo pertence aos
paradigmas de inferência pesada — TabPFN 7.416, STab 8.645 e TabFM
43.262 µs/linha. A fronteira de Pareto custo×desempenho da binária é
{TabPFN, XGBoost, LightGBM}: todo o aprendizado profundo clássico é
dominado. A inversão do foundation model — treino de segundos, serving
proibitivo — é o achado-âncora que motiva o Capítulo 7.

## 5.4 Ato III — reversões condicionais e o eixo do tamanho

A melhor família depende da estrutura do problema. Na multiclasse (N=3,
descritivo), o DL da geração 2024 assume o topo clássico; sob
desbalanceamento severo, os GBDTs ganham vantagem relativa; no extremo
categórico, o one-hot penaliza a atenção; e o TabPFN combina o melhor rank
médio com o melhor pior-caso (raramente catastrófico). As curvas de
aprendizado (6 modelos × 5 conjuntos × 3 sementes, 500→pool) dão forma ao
eixo do tamanho: o cruzamento de liderança GBDT↔DL é real e específico por
conjunto — em 3 de 5, o DL ultrapassa e não devolve; em adult, o GBDT
retoma em n≈4.000 — e o TabPFN é o melhor modelo em *todos* os conjuntos
no regime n≤4.000, quantificando o nicho de poucos dados do paradigma
zero-shot.

## 5.5 Ato IV — a quebra do empate (geração 2026)

A expansão para 14 modelos reorganiza a narrativa. O TabFM lidera o rank
médio nas três tarefas — 2,7 na binária, 1,0 (perfeito) na multiclasse,
1,2 na regressão — e é o melhor modelo absoluto em 13 dos 18 conjuntos,
participando de 17. O empate do Ato I fica corretamente re-escopado como
retrato da geração anterior: não foi o DL tunado que alcançou os GBDTs;
foi o pré-treino que atropelou ambos. A cautela estatística permanece
dita: com k=14 e N=10, o CD de Nemenyi é largo — a separação do TabFM
apoia-se na consistência entre conjuntos e no bayesiano, não no post-hoc.

No mesmo movimento, o teste independente das KANs produz o resultado
negativo de valor: KAN (11,0/12,0/9,8) e TabKAN (11,4/13,3/11,8) ocupam o
último terço nas três tarefas, o TabKAN atrás do TabNet em multiclasse e
regressão — sob protocolo uniforme, as alegações dos artigos originais não
se sustentam, e a literatura cética é confirmada. Registre-se a contraparte
de engenharia: ambos habitam o tier rápido de latência (2–3 µs/linha) e
custam VRAM desprezível — o problema é o desempenho, não o custo.

## 5.6 Síntese do benchmark

Três fatos encadeados emergem: (i) na geração até 2025, o desempenho bruto
empata no topo e a decisão migra para custo, latência, robustez e
estrutura — o arcabouço multicritério do Capítulo 6; (ii) a geração 2026
quebra o empate em acurácia ao preço de 4-5 ordens de magnitude em
latência — tornando o arcabouço *mais* necessário e apontando seu único
obstáculo; (iii) o obstáculo é atacável — o Capítulo 7 investiga quanto
dele a destilação remove, e sob quais condições verificáveis.
