# Fala pro professor — texto corrido, informal e direto (estado: 14 modelos)

> Pra ler falando numa conversa 1-a-1 com o orientador. As marcações
> *(em itálico)* são dicas de quando abrir a figura — não se lê em voz alta.
> Números conferidos contra `TABELA_RESULTADOS.md`, o deck e `paper_retention.csv` em 21/07/2026.

---

Então, professor, deixa eu te atualizar — desde a última conversa o benchmark
cresceu pros quatorze modelos que combinamos, e o resultado mudou a história de
um jeito que eu acho que o senhor vai gostar.

Relembrando a espinha: a frase "em dado tabular, boosting sempre vence" foi
formada antes da leva nova. Eu montei o benchmark pra re-testar isso direito —
nested cross-validation, tuning bayesiano igual pra todo mundo, teste
estatístico frequentista e bayesiano, quatorze modelos em dezoito datasets. Dos
quatorze, sete vieram de biblioteca oficial, um da comunidade, um é backbone
vendorizado com adaptações documentadas, e cinco eu implementei do zero a
partir dos papers — incluindo os dois novos da família Kolmogorov-Arnold, o KAN
e o TabKAN. E os dois foundation models: o TabPFN e o TabFM do Google, que saiu
há três semanas — a gente deve estar entre os primeiros do mundo a medi-lo num
protocolo neutro.

A história agora tem dois movimentos.

**Movimento um: o empate — que era a manchete até aqui — vale para a geração
até 2025.** Entre boosting, deep learning e o TabPFN, o topo é estatisticamente
indistinguível: pelo teste bayesiano com zona de indiferença de um ponto de
AUC, oito dos dez modelos são praticamente equivalentes ao líder. E eu fiz a
análise de sensibilidade desse limiar, que era uma pendência de rigor: com
meio ponto, sobram XGBoost e CatBoost equivalentes; com dois pontos, todos. Ou
seja, a conclusão qualitativa aguenta o aperto. E oitenta por cento da variância
de rank é interna às famílias — o modelo importa mais que a família.

*(mostrar a §6.2 do deck)*

**Movimento dois: a geração 2026 quebra o empate.** O TabFM, zero-shot, sem um
segundo de tuning, é o número um absoluto em **treze dos dezoito datasets** — e
lidera o rank médio nas **três** tarefas: 2.7 na binária, 1.0 perfeito na
multiclasse, 1.2 na regressão. Não foi o deep learning tunado que alcançou o
boosting, como a literatura de 2024 sugeria; foi o pré-treino que atropelou os
dois. Só que tem o contrapeso, e ele é brutal: a inferência custa 43 milissegundos
por linha no adult, sob o mesmo protocolo de latência dos outros modelos —
131 mil vezes o XGBoost, e mais ainda nos contextos maiores. O melhor modelo em acurácia é o pior em latência. Então a pergunta de
decisão mudou de "qual família?" para "**a latência do foundation model cabe no
meu caso de uso?**" — e é exatamente isso que o nosso framework multicritério
responde.

Dois achados menores que valem menção. Primeiro, as KANs: fui rigoroso na
integração — sanity check contra o paper, desvios de protocolo documentados — e
no teste independente elas afundaram: meio de tabela na binária, último terço
nas três tarefas, o TabKAN atrás até do TabNet em multiclasse e regressão. O
paper delas comparava contra um XGBoost fraco; sob protocolo uniforme, a
alegação não se sustenta. É resultado negativo, mas é o primeiro teste neutro
da família que a literatura ganha — isso tem valor.

Segundo, eu validei o framework de decisão de um jeito que ele ainda não tinha
sido: leave-one-dataset-out. A árvore de meta-features, como preditor, **não**
generaliza — acerto abaixo do baseline trivial, e eu digo isso com todas as
letras. Mas a política "**foundation model primeiro, desvie só por
restrição**" tem regret mediano **zero** com os quatorze modelos. Ou seja: o
flowchart, que sempre foi orientado a restrições — latência? tamanho? número de
classes? — sai da validação fortalecido. As perguntas certas eram as
restrições, não adivinhar o vencedor.

E aí chega a parte da contribuição, onde eu fui atrás do rigor máximo antes de
gastar um mês de GPU. A ideia era destilar o foundation model num aluno rápido
— pegar a acurácia do TabFM e servir com latência de XGBoost. Fiz a checagem de
novidade datada, e descobri que a formulação básica **já foi publicada em maio
deste ano** — um grupo destilou foundation models em árvores para
**classificação**, em 153 datasets. Se eu tivesse construído sem checar, seria
refutado na primeira revisão. Mas o recorte que sobrou é forte e é nosso: a
**regressão distribucional** ninguém fez — os foundation models preveem
distribuições na regressão, não pontos, e destilar a distribuição inteira num
GBDT de quantis, medindo com CRPS e calibração de intervalos, é gap aberto.
Registrei o desenho antes de rodar: hipóteses com números, controles, critérios
de go/no-go. E executei o programa inteiro — seis fases, vinte datasets, que é
o pool elegível completo da suíte CTR23 sob regra fixada antes de qualquer
resultado. O que saiu: a destilação **pontual** não funciona em escala
realista — refutei a minha própria hipótese, com dois teachers diferentes. Mas
a **distribucional** — transferir as curvas de quantis do teacher — funciona:
positiva em dezesseis dos vinte datasets, retenção mediana de dezenove por
cento, chegando a um caso em que o aluno *supera* o próprio teacher. E com
significância nos dois instrumentos: teste de sinal p=0,006, Wilcoxon p=0,045,
mais setenta por cento de massa posterior no bayesiano. Também descobri quando
NÃO destilar — virou uma regra de decisão prática: transforme o alvo e teste
um aluno nativo forte primeiro; destile quando a vantagem do teacher
sobreviver a isso. O preprint está com o draft completo, duas figuras
validadas, e eu quero submeter até outubro, dentro do cronograma.

Resumindo numa frase: a regra de que boosting sempre vence não é falsa — ela
expirou. Valia até 2025; em 2026 o pré-treino quebrou o empate, o custo virou o
eixo da decisão, e a nossa contribuição ataca exatamente o preço que o novo
líder cobra. Benchmark completo, framework validado, contribuição executada e
com draft de preprint pronto pra sua revisão. É isso — o que o senhor acha?
