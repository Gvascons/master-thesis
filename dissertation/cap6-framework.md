# Capítulo 6 — O Arcabouço de Decisão Validado

> Rascunho v1 (19/07/2026). Fontes: notebooks 08-09, deck §6.2/§7-8,
> `scripts/lodo_validation.py`, `results/aggregated/lodo_validation{,_14}.csv`.

## 6.1 Do descritivo ao validado

O benchmark multicritério do Capítulo 5 produz, por construção, um artefato
prático: dado um problema novo, que modelo usar? A primeira versão do
arcabouço respondia com dois instrumentos descritivos — uma matriz
multicritério (desempenho, custo de treino, latência, robustez,
interpretabilidade) e um fluxograma de três perguntas orientado a
restrições. Este capítulo eleva o arcabouço de descritivo a **validado**,
com dois resultados complementares: um negativo (o que o arcabouço *não*
deve tentar fazer) e um positivo (a política que os dados sustentam).

## 6.2 O resultado negativo: roteamento por meta-features não generaliza

A literatura de meta-learning sugere prever o algoritmo vencedor a partir de
meta-features do conjunto (tamanho, dimensionalidade, desbalanceamento,
fração categórica). Testamos essa promessa com validação
*leave-one-dataset-out*: uma árvore de profundidade 2 (o mesmo modelo
ilustrativo do Capítulo 5) treinada nos demais 17 conjuntos prevê a família
vencedora do conjunto excluído.

Resultado: **hit-rate de 0,11 contra 0,44 do baseline de classe
majoritária** (dados de 11 modelos). Com N=18 conjuntos, aprender o
roteamento é estatisticamente inviável — e afirmar isso com um protocolo
explícito protege o arcabouço da crítica de overclaiming que atinge parte da
literatura de recomendação de algoritmos. A árvore permanece no trabalho
como *ilustração descritiva da estrutura* das reversões, jamais como
preditor.

## 6.3 O resultado positivo: a política "FM primeiro" e sua evolução

Como alternativa ao roteamento aprendido, avaliamos políticas fixas pela
métrica de **regret normalizado** (0 = a família recomendada contém o melhor
modelo do conjunto; 1 = contém apenas o pior):

| Política | Regret médio @11 | mediano @11 | médio @14 | mediano @14 |
|---|---|---|---|---|
| Árvore (LODO) | 0,228 | 0,145 | 0,051 | 0,000 |
| Sempre-GBDT | 0,249 | 0,168 | 0,305 | 0,291 |
| Sempre-DL | 0,189 | 0,132 | 0,313 | 0,349 |
| **Sempre-FM** | **0,115** | **0,018** | **0,021** | **0,000** |

Já na geração ≤2025, "usar o foundation model por padrão" era a melhor
política fixa. Com a geração 2026 (TabFM), o regret mediano da política cai
a **zero**: em metade ou mais dos conjuntos, a família FM contém o melhor
modelo, ponto. A pergunta "qual família vence?" se dissolve — e com ela o
valor de qualquer roteador de desempenho.

## 6.4 O arcabouço resultante: decisão por restrições

O que resta — e o que o fluxograma sempre codificou — é a decisão por
**restrições verificáveis a priori**:

1. **Latência/custo de inferência.** O líder de acurácia custa 4-5 ordens de
   magnitude mais para servir (Cap. 5); se o orçamento de latência é de
   microssegundos, a família FM está excluída na forma nativa — e o Capítulo
   7 investiga recuperá-la por destilação.
2. **Limites arquiteturais.** >10 classes (TabFM) ou >limite de contexto:
   exclusões estruturais objetivas.
3. **Interpretabilidade intrínseca exigida** (regulação): GBDTs.
4. **Nenhuma restrição ativa:** foundation model, com o TabPFN como opção
   madura e o TabFM como fronteira (com as ressalvas de maturidade do §5.x).

A validação por regret confirma que essas quatro perguntas capturam o
essencial: as restrições são o *conteúdo informativo real* da decisão, e o
desempenho bruto deixou de discriminar dentro da família líder.

## 6.5 Limitações

N=18 impede validação estatística fina do próprio arcabouço (o LODO é a
ferramenta honesta possível); o regret é calculado sobre a métrica primária,
não sobre combinações multicritério ponderadas por preferências do usuário
(extensão natural); a fronteira de modelos muda em ciclos de meses — o
arcabouço é datado por construção e versionado com o benchmark que o
sustenta.
