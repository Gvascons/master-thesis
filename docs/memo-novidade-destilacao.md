# Memorando — verificação de novidade: destilação de foundation models tabulares

> Verificação online conduzida em 15/07/2026, ANTES de investir na
> contribuição. Conclusão executiva: **a formulação básica da ideia já foi
> publicada (mai/2026); prosseguir com recorte ajustado** — regressão
> distribucional + fronteira de Pareto multi-critério + TabFM como teacher.

## O que já existe (prior work a citar desde o dia 1)

- **Pocket Foundation Models** (Tanna et al., arXiv 2605.18654, mai/2026;
  aceito no 2º ICML Workshop on FMs for Structured Data): destilação
  por-dataset de TFMs (TabICLv2, TabPFNv2.6, LimiX, Orion-MSP) em
  XGBoost/CatBoost/MLP via soft labels (KL com temperatura; para árvores,
  regressão MSE por classe nos logits), com **rotulagem out-of-fold** para
  evitar colapso dos soft targets (teacher ICL pontuando o próprio treino
  produz rótulos quase one-hot). 153 datasets de **classificação**; aluno
  retém 96,5% do AUC a 1,9 ms em CPU (38–860× speedup). Cobre também a
  análise "quando funciona vs falha" (ganhos concentrados em baixa
  dimensionalidade).
- **Paper-irmão em saúde** (arXiv 2605.18702): 19 datasets clínicos,
  calibração e fairness; multi-teacher não supera o melhor teacher único.
- **Prior Labs / TabPFN-2.5** (arXiv 2511.08667, nov/2025): "distillation
  engine" **comercial fechado** (TabPFN → MLP/árvores por dataset) — sem
  metodologia nem benchmark públicos.
- **TabDistill** (arXiv 2511.05704): TabPFN→MLP minúsculo, mas só few-shot;
  nome ocupado.
- **Destilação de TFM em GAMs** (arXiv 2604.13332): para interpretabilidade,
  não velocidade.
- **Aceleração NÃO-destilativa** (concorrência da motivação): TACO
  (compressão de contexto ~1%, até 94×, ICML 2026, arXiv 2602.05649),
  MotherNet (hipernetwork gera MLP em um forward, arXiv 2312.08598), TabFlex,
  TabICLv2, KV-caching do contexto.
- **Mecânica de soft labels em árvores é literatura estabelecida**: Bucila,
  Caruana & Niculescu-Mizil 2006 (model compression + transfer set MUNGE);
  Breiman & Shang 1996; Born-Again Tree Ensembles (Vidal & Schiffer, ICML
  2020); Frosst & Hinton 2017.

## O que está aberto (nosso recorte)

1. **Regressão distribucional (pilar principal).** Nenhum trabalho público
   destila TFM→aluno rápido para regressão. Não-trivial: TabPFN v2.5 e TabFM
   produzem **distribuições preditivas** na regressão — destilar a
   distribuição (não só a média) em GBDT via quantile regression / alvos
   distribucionais, avaliando com CRPS e cobertura de intervalos, é gap real.
2. **Fronteira de Pareto multi-critério** (acurácia × latência × memória ×
   custo total) comparando **três estratégias**: destilar vs comprimir
   contexto (TACO) vs cachear — inédito; nosso benchmark neutro 14×18 é o
   substrato perfeito.
3. **TabFM (Google, jun/2026) como teacher** — seríamos os primeiros
   (ornamento com janela curta, não pilar).
4. **Transfer set aumentado** (gerar pontos extras com o teacher/TabPFGen,
   linha Bucila 2006) — Pocket FM usa só o pool de treino.

## Decisões metodológicas herdadas

- Adotar a **rotulagem out-of-fold** do Pocket FM como baseline (é o jeito
  certo; reinventar seria pior e ignorá-lo seria vazamento).
- Não usar o nome "TabDistill" (ocupado).
- Não reivindicar "primeira destilação de FM tabular" em hipótese alguma.

## Título de trabalho (proposta)

*"Destilação de foundation models tabulares além da classificação: regressão
distribucional e a fronteira de Pareto entre destilar, comprimir e cachear."*

## Urgência

A área anda em ciclos de ~3 meses (TabPFN-2.5 nov/25 → Pocket FM mai/26 →
TabFM jun/26). Executar o recorte de regressão + preprint idealmente até
**outubro/2026** — compatível com o cronograma da prorrogação (contribuição
em set-out, redação em nov).


## Re-varredura de 20/07/2026 (pré-redação)

Recorte segue aberto. Novos itens mapeados: TL-ANDI (arXiv 2607.04809 —
transferência entre tarefas com rótulos destilados; ortogonal, citar),
CRUMB (arXiv 2606.11473 — compressão de contexto via MMD; baseline
representativo do braço "comprimir"), TabPFN-3 existe (arXiv 2605.13986 —
justificamos v2.5 por pinagem/hardware). Pocket FM segue v1
(classificação apenas); Prior Labs adquirida pela SAP (17/07, sem anúncio
técnico novo). Buscas dirigidas por distillation+quantile/CRPS tabular:
zero colisões.
