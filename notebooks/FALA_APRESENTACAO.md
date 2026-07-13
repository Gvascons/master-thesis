# Fala pro professor — texto corrido, informal e direto

> Pra ler falando numa conversa 1-a-1 com o orientador. Tom direto, sem
> formalidade de plateia. As marcações *(em itálico)* são só dica de quando abrir
> a figura — não se lê em voz alta.

---

Então, professor, deixa eu te mostrar o que eu fiz. A ideia toda parte de uma frase
que todo mundo repete: "em dado tabular, boosting sempre vence". Boosting é o
XGBoost, LightGBM, CatBoost — esses modelos que empilham centenas de arvorezinhas,
cada uma corrigindo o erro da anterior. A crença é que eles ganham do deep learning e
acabou. Só que essa frase é de antes da leva nova de 2024-2025: redes feitas
sob medida pra dado tabular, tipo TabM e RealMLP, e os *foundation models*, o TabPFN
à frente — um modelo que já vem pré-treinado e você usa em modo *zero-shot*, sem
treinar nada. Então minha pergunta é: essa frase ainda vale? E se não, o que muda?

Pra testar isso direito eu montei um benchmark grande — onze modelos em dezoito
datasets do OpenML, cobrindo os três tipos de tarefa: dez de classificação binária,
três de multiclasse e cinco de regressão, de mil a quinhentas e oitenta mil amostras.
Um detalhe honesto: seis modelos vieram de biblioteca oficial, mas quatro eu
implementei do zero em PyTorch a partir do paper — incluindo o SAINT e o STab, que
não têm pacote maduro. Foi bastante trabalho de engenharia, e é justo registrar
porque, se um desses vai mal, fica a dúvida se é o modelo ou a minha implementação.

*(esquema do protocolo)*

Sobre medir de forma justa, o esqueleto é validação cruzada aninhada. Rapidão: um
laço externo só pra medir o modelo honestamente, e dentro dele um laço interno só pra
escolher os hiperparâmetros — os "botões" de cada algoritmo. O pulo do gato é que a
escolha dos botões nunca vê o teste, então eu não tenho vazamento de dado, que é o
modelo parecer melhor do que é porque espiou a resposta sem querer. Essa busca dos
botões eu faço com o Optuna, que é uma otimização que aprende onde vale a pena
procurar em vez de testar no chute.

Agora os resultados. Eu penso em três atos porque tem uma virada.

**Ato um: o empate.** No desempenho puro, o topo é estatisticamente indistinguível.
Eu rodei o teste de Friedman — que pergunta "tem diferença real ou é só ruído?" — e
ele diz que tem. Mas o teste seguinte, o *post-hoc*, que é o que localiza onde estão
as diferenças, não separa o pelotão de cima: tá todo mundo amontoado.

*(average_ranks e cd_diagram_binary)*

E aqui já aparece a primeira pista: o TabPFN lidera binária e regressão, mas na
multiclasse quem lidera é o deep learning — o vencedor já depende da tarefa. Nesse
diagrama de diferença crítica, todo mundo ligado pela barra grossa é empatado, e o
topo é um bloco só. Pra não confiar num teste só, reforcei o empate de dois jeitos:
um teste bayesiano, que em vez de "sim ou não" me dá a probabilidade de um ser melhor
que o outro dentro de uma margem que na prática não importa — e por essa lente, oito
dos dez modelos são praticamente equivalentes ao TabPFN. E uma decomposição de
variância, que mostrou que oitenta por cento da variação de desempenho está *dentro*
das famílias, não entre elas. Ou seja: "boosting versus deep learning" é uma
abstração fraca, o modelo específico importa muito mais que a família. O único sinal
robusto é lá no fundo — o TabNet é sempre o pior.

**Ato dois: o custo.** Se todo mundo empata em acurácia, a decisão migra pra outro
eixo. E esses modelos que empatam são absurdamente diferentes em custo: o treino
varia trezentas vezes, e a latência de inferência varia vinte e seis mil vezes.

*(pareto_binary e inference_time)*

Esse gráfico é a fronteira de Pareto — quem está nela é porque não existe ninguém ao
mesmo tempo mais barato e melhor. E a fronteira é TabPFN, XGBoost e LightGBM; todo o
deep learning fica dominado, sempre tem alguém mais barato e igual ou melhor. E o
achado que eu considero a âncora é a inversão do TabPFN: ele é o mais barato pra
treinar — um segundo, zero tuning, porque é zero-shot — mas é o segundo mais caro pra
usar, umas vinte e duas mil vezes mais lento que o XGBoost na inferência. Vira a
lógica de cabeça pra baixo: de graça pra treinar, caríssimo pra pôr em produção.

**Ato três: as reversões.** É onde a frase original desmorona de vez — a melhor
família depende da estrutura do problema.

*(cd_diagram_multiclass e robustness_riskmap)*

Na multiclasse, o deep learning sobe pro topo (STab, TabM, RealMLP) — a reversão mais
marcante, mas com a ressalva de serem só três datasets, então é descritivo. Sob dados
muito desbalanceados, tipo fraude, o boosting se sai relativamente melhor. Com muita
variável categórica, os modelos de atenção engasgam porque a codificação explode de
tamanho. E esse mapa de risco mostra uma coisa legal do TabPFN: ele tem o melhor
desempenho médio e o melhor pior-caso ao mesmo tempo — quase nunca é catastrófico,
enquanto todos os outros têm pelo menos um dataset onde despencam.

*(learning_curves — a parte nova)*

E essa aqui é a parte que eu fiz depois e que amarra tudo. Repara que a briga
original — "boosting ganha" contra "o deep learning alcançou" — é no fundo uma
questão de quantos dados você tem. Então eu fui medir de frente: peguei cinco
datasets e re-treinei cada modelo em fatias crescentes, de quinhentas amostras até o
conjunto todo, pra ver onde a liderança troca de mãos. E em três dos cinco, o deep
learning começa atrás e ultrapassa o boosting conforme o dado cresce, sem devolver a
liderança; num deles o boosting retoma lá pelas quatro mil amostras; e num deles fica
boosting o tempo todo. Não tem resposta única, o ponto de cruzamento é específico de
cada problema. Mas o número mais limpo é o do regime de poucos dados: até quatro mil
amostras, o TabPFN é o melhor nos cinco datasets, sem exceção. É a prova quantitativa
de que os foundation models mudaram o jogo quando você tem pouco dado — e casa com o
custo do ato dois: barato de treinar e o melhor justo onde o dado é escasso.

*(decision_matrix e decision_flowchart)*

Juntando tudo, a contribuição concreta — que não existe na literatura pra essa leva
de 2024-2025 — é um framework de decisão. Uma matriz que pontua cada modelo em cada
critério, e a mensagem é que ninguém tira nota máxima em tudo: o TabPFN domina
desempenho e robustez mas é fraco em latência, e o boosting é o generalista
equilibrado. E um fluxograma prático: o cara responde três perguntas — precisa de
resposta rápida? qual o tamanho e a dimensão? tem muito categórico? — e sai com uma
lista curta e defensável de modelos pra testar.

Fechando, e sendo transparente nos limites: meu poder estatístico é baixo, são dez,
três e cinco datasets por tarefa, e por isso eu não cravo "fulano é o vencedor" —
sustento o empate com o bayesiano e a decomposição de variância, que são honestos
sobre a incerteza. E o mais importante: pra onde isso aponta. O gargalo mais concreto
que apareceu foi como o deep learning lida com variável categórica — aquela explosão
que estoura a memória da GPU. Minha próxima etapa nasce daí: um jeito melhor de
representar categórico com *embeddings* aprendidos, pra fechar a distância pro
boosting nesse tipo de dado. E o bom é que agora eu tenho como medir sucesso de forma
objetiva — é mover aquela curva de aprendizado pra esquerda, fazer o deep learning
alcançar o boosting com menos dado. Não é um "melhorar" vago, é um alvo num gráfico
que eu já sei desenhar.

Resumindo numa frase: a regra de que boosting sempre vence não é falsa, é
incompleta. Ela vale numa foto, e o que eu mostro é o filme — quando a acurácia
empata, e ela empata, a decisão esperta migra pro custo, pra latência, pra estrutura
do problema e pro tamanho do dado. É isso que transforma "qual o melhor modelo?" em
"melhor pra quê, com quantos dados, e a que custo?". É isso, o que você acha?
