# Capítulo 3 — Trabalhos Relacionados

> Prosa v1 (20/07/2026). Cada eixo fecha com o posicionamento explícito
> deste trabalho. Verificações de novidade datadas em
> `docs/memo-novidade-destilacao.md` (13-20/07/2026).

## 3.1 Benchmarks de modelos tabulares

Grinsztajn et al. [2022] deram forma quantitativa ao ditado "árvores vencem
em tabelas", identificando vieses indutivos (irregularidade das funções
alvo, atributos não informativos) que favorecem árvores sobre redes.
McElfresh et al. [2024] matizaram: em larga escala de conjuntos, redes
vencem com frequência não trivial, e meta-características predizem
parcialmente quando. Gorishniy et al. [2021] padronizaram protocolos de
comparação para o DL tabular. Mais recentemente, o TabArena [2025] mantém
um benchmark "vivo" com dezenas de conjuntos e ranking Elo.

**Posicionamento.** Nosso benchmark não compete em largura (18 conjuntos <
TabArena); diferencia-se em três frentes: (i) profundidade de protocolo por
célula — nested CV com tuning uniforme por família, testes frequentistas
*e* bayesianos com análise de sensibilidade; (ii) multicritério — latência,
custo de treinamento, robustez e interpretabilidade medidos sob o mesmo
protocolo, e não apenas acurácia; (iii) atualidade verificada — inclusão
simultânea da geração 2026 de foundation models (TabFM, medido semanas após
o lançamento) e da família KAN, ambas ausentes dos benchmarks neutros à
data das nossas verificações (13-15/07/2026).

## 3.2 A controvérsia GBDT vs. aprendizado profundo

As duas narrativas — "árvores vencem" [Grinsztajn et al., 2022] e "o DL
moderno alcançou" [Holzmüller et al., 2024; Gorishniy et al., 2025] — são
frequentemente tratadas como contraditórias. Nossa leitura, sustentada
pelas curvas de aprendizado do Capítulo 5, é que ambas são afirmações
condicionais a tamanho amostral e estrutura da tarefa, verdadeiras em
regimes distintos: o cruzamento de liderança é real, específico por
conjunto, e mensurável quando se varre o tamanho do treino diretamente. A
chegada dos foundation models desloca a controvérsia inteira: o novo eixo
não é árvore vs. rede, mas pré-treinado vs. treinado do zero.

## 3.3 Avaliação de KANs em dados tabulares

Os proponentes reportam vantagens: o TabKAN [Eslamian et al., 2025] alega
superar XGBoost e transformers tabulares em dez conjuntos; o TabKANet
[Gao et al., 2024] usa KAN como módulo de embedding. A literatura crítica
aponta o contrário: sob paridade de parâmetros e FLOPs, o MLP domina fora
de regressão simbólica [Yu et al., 2024]; benchmarks independentes de
pequena escala mostram resultados mistos com custo computacional maior
[Poeta et al., 2024]. Nenhum benchmark tabular neutro de larga escala
incluía KANs até nossas verificações.

**Posicionamento.** Fornecemos o teste independente que faltava, sob
protocolo uniforme e com sanity checks contra os valores publicados. O
resultado (Capítulo 5) é desfavorável às alegações fortes: KAN e TabKAN
terminam no último terço nas três tarefas (ranks médios 11,0/12,0/9,8 e
11,4/13,3/11,8), com o TabKAN atrás do próprio TabNet em multiclasse e
regressão — evidência de que os baselines sub-tunados dos artigos originais
explicam as alegações, e confirmação em protocolo neutro da literatura
crítica.

## 3.4 Foundation models tabulares: avaliação e aceleração

A linhagem PFN/ICL (TabPFN v2/v2.5/3, TabICL, TabFlex, MotherNet) e o
TabFM [Google, 2026] foram descritos no Capítulo 2. Sobre avaliação: os
números públicos do TabFM à data do lançamento provinham do próprio Google
(TabArena) com verificação independente apenas parcial. Sobre aceleração
sem destilação: TACO comprime o contexto a ~1% com speedups de até 94×;
CRUMB seleciona contexto compacto via MMD sem retraining; MotherNet
amortiza o ajuste num hipernetwork que emite pesos de MLP; TL-ANDI combina
rótulos destilados e seleção de contexto por transporte ótimo para
transferência entre tarefas; e a engenharia simples — truncar contexto,
reduzir ensemble, cache — é folclore não medido.

**Posicionamento.** (i) Até onde verificamos, a primeira comparação neutra
TabPFN × TabFM sob protocolo idêntico, enquadrada como replicação
inter-laboratórios do paradigma; (ii) a primeira medição *lado a lado* das
estratégias de aceleração — destilar, comprimir contexto, reduzir ensemble
— nos mesmos hold-outs (a fronteira do Capítulo 7), com dois achados
próprios (ensemble de 1 membro ≈ 8 a 1/7 da latência; o último passo de
contexto custa 3,3× de latência por ganho marginal em GPUs de 16 GB).

## 3.5 Destilação de foundation models tabulares

O Pocket FM [Tanna et al., 2026] estabelece a formulação para
classificação: alunos XGBoost/CatBoost/MLP treinados em soft labels OOF de
múltiplos teachers, 153 conjuntos, retenção de 96,5% do AUC com speedups de
38–860×; o companion clínico [2026] adiciona calibração e fairness — ambos
exclusivamente em classificação. A Prior Labs comercializa um engine de
destilação fechado para o TabPFN-2.5 (sem metodologia pública). TabDistill
[2025] cobre o regime few-shot; outra linha destila TFMs em GAMs para
interpretabilidade. Verificações datadas (13-20/07/2026, incluindo buscas
dirigidas por destilação+quantis/CRPS): **nenhum trabalho público cobre
regressão, saídas distribucionais ou uma fronteira entre estratégias de
serving**.

**Posicionamento (a lacuna deste trabalho).** Regressão distribucional
(transferência de curvas de quantis, avaliada por CRPS/cobertura) — aberta;
fronteira de três estratégias — aberta; TabFM como teacher — aberta (e
fechada por nós com um negativo arquitetural: a cabeça de regressão do
TabFM é pontual, e o spread do ensemble é mensuravelmente inútil como
distribuição). Adotamos o OOF do Pocket FM como base e delimitamos nossa
contribuição nesse eixo ao que ele protege *em regressão*: calibração, não
acurácia.

## 3.6 Meta-learning para seleção de modelo

A recomendação de algoritmo por meta-features tem literatura longa e
resultados historicamente frágeis fora de grandes coleções. Com N=18
conjuntos, a honestidade exige validação explícita: nossa análise
leave-one-dataset-out (Capítulo 6) mostra que a árvore de meta-features
não supera baselines triviais como preditor (hit-rate 0,11 vs. 0,44), e
que a política fixa "foundation model primeiro" domina por regret. O
arcabouço resultante é deliberadamente *constraint-driven* — decidir por
restrições verificáveis, não por roteamento aprendido — uma posição
metodológica que registramos como contribuição de honestidade frente à
tentação de sobreajustar recomendadores a poucas observações.
