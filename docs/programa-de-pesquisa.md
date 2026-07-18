# Programa de pesquisa — arquitetura das entregas do mestrado

> Documento-mestre de planejamento e rastreabilidade. Formaliza as três
> entregas (AI-1, AI-2, dissertação), o fio condutor científico, o estado de
> cada frente, os critérios de conclusão e os riscos. Versionado no
> repositório para que nada dependa de estado local. Campos entre [ ] exigem
> confirmação do aluno/secretaria — não foram assumidos.
> Última revisão: 18/07/2026 (benchmark 14×18 completo; AI-1 consolidada;
> LODO@14 concluído; piloto de destilação em execução).

## 0. Fio condutor científico (a tese em um parágrafo)

Em dados tabulares, o ditado "GBDT sempre vence" foi formado antes da geração
2024-2026 de deep learning tabular e dos foundation models. Nosso benchmark
neutro (14 modelos × 18 datasets, nested CV, testes frequentistas e
bayesianos) mostra três fatos encadeados: (1) entre os modelos clássicos e
DL, o topo é um empate estatístico — e a decisão migra para custo, latência,
robustez e estrutura da tarefa; (2) a geração 2026 de foundation models
(TabFM) quebra esse empate em acurácia, mas ao custo de latência ~5 ordens de
magnitude acima dos GBDTs — tornando o arcabouço multicritério MAIS
necessário, não menos; (3) o obstáculo que impede a política ótima validada
("foundation model primeiro") é removível por destilação — e a fronteira
aberta é a destilação **distribucional em regressão**, nossa contribuição.

## 1. Entrega 1 — TRABALHO INDIVIDUAL EM INTELIGÊNCIA COMPUTACIONAL 1 (AI-1)

**Escopo:** o benchmark multicritério e sua leitura (os "4 atos": empate →
custo → reversões/curvas de aprendizado → framework de decisão).

**Formato/prazo:** [confirmar com a secretaria: formato do documento, data de
entrega e de apresentação]. Material de apresentação já pronto.

**Estado: ~99% — núcleo completo (18/07/2026).**

| Item | Estado | Onde |
|---|---|---|
| Benchmark **14 modelos** × 18 datasets (250/252; 2 exclusões estruturais no helena) | ✅ | `results/raw`, `results/aggregated` |
| Protocolo estatístico auditado + sensibilidade do ROPE | ✅ | `docs/metodologia-estatistica.md`, `rope_sensitivity.csv` |
| Curvas de aprendizado / crossover (RQ2) | ✅ | `notebooks/10_learning_curves.ipynb` |
| Notebooks 01-09 re-executados @14 + figuras | ✅ | `notebooks/`, `results/figures/` |
| Deck @14 (narrativa em 2 movimentos, §6.2 nova) + tabela mestra | ✅ | `notebooks/00_presentation.ipynb`, `TABELA_RESULTADOS.md` |
| Roteiros de fala atualizados @14 | ✅ 18/07 | `ROTEIRO_APRESENTACAO.md`, `FALA_APRESENTACAO.md` |
| Latência dos 3 modelos novos | ⏳ (~10 min de GPU; após piloto de destilação) | `scripts/measure_latency.py` |

**Achados-síntese do benchmark @14:** TabFM lidera as três tarefas (rank
médio 2,7/1,0/1,2) e é o nº 1 absoluto em 13/18 datasets, ao custo de
latência ~5 ordens acima dos GBDTs; KAN/TabKAN ficam no último terço nas
três tarefas (teste independente desfavorável às alegações do TabKAN);
o "empate" do estudo original fica corretamente re-escopado como retrato da
geração ≤2025.

## 2. Entrega 2 — ATIVIDADE DE ORIENTAÇÃO INDIVIDUAL (AI-2)

**Escopo:** a contribuição focada, em dois pilares complementares que saem do
benchmark:

**Pilar A — Destilação distribucional de foundation models (a aposta de
estado da arte).** Recorte validado por checagem de novidade
(`docs/memo-novidade-destilacao.md`): regressão distribucional (inédita),
fronteira de Pareto destilar-vs-comprimir-vs-cachear, TabFM como teacher.
Desenho pré-registrado com hipóteses falsificáveis
(`docs/desenho-experimental-destilacao.md`); harness implementado e validado
em smoke test (`scripts/distill.py`), com os sinais qualitativos já na
direção das hipóteses (ponto: destilado 0.610 vs controle 0.636 RMSE;
distribucional: CRPS −11%, calibração 0.73 vs 0.36 de PICP80). Meta:
**preprint até outubro/2026** (a área anda em ciclos de ~3 meses).

**Pilar B — Framework de decisão validado (o produto). CONCLUÍDO em
essência (18/07):** (i) resultado negativo rigoroso — roteamento por
meta-features não generaliza sob LODO (hit 0.11 vs baseline 0.44 no estudo
@11); (ii) política "FM primeiro, desvie por restrição" com regret mediano
0.018 @11 e **0.000 @14** (`lodo_validation_14.csv`) — com a geração 2026 a
questão do roteamento por desempenho se dissolve e restam as restrições;
(iii) matriz multicritério e flowchart regenerados com 14 modelos. Resta
apenas a redação do relatório consolidado (com o Pilar A).

**Formato/prazo:** [confirmar formato exigido; prazo interno: alinhado ao
cronograma da prorrogação — set/out 2026].

**Critério de conclusão:** piloto de destilação nos pools completos
executado e decidido pelo gate go/no-go pré-registrado; grade completa nos 5
datasets de regressão (+ extensão OpenML-CTR23 se go); LODO re-executado com
14 modelos; relatório técnico consolidando os dois pilares; preprint
submetido (meta-stretch) ou pronto para submissão.

## 3. Dissertação (conclusão nov/2026, defesa dez/2026)

**Esqueleto de capítulos** (rascunho de arquitetura; material-fonte já
existente indicado):

1. **Introdução** — motivação, perguntas de pesquisa, contribuições.
2. **Fundamentação** — GBDT, DL tabular, foundation models/ICL, destilação.
   (fontes: memorandos de pesquisa em `docs/`)
3. **Trabalhos relacionados** — benchmarks (Grinsztajn, McElfresh, TabArena),
   modelos 2024-26, destilação de TFMs (Pocket FM et al.).
4. **Metodologia experimental** — protocolo, métricas, testes.
   (fonte: `docs/metodologia-estatistica.md` — praticamente pronto)
5. **Benchmark multicritério** — resultados dos 4 atos com 14 modelos.
   (fonte: notebooks 01-10 + deck)
6. **Framework de decisão validado** — matriz, flowchart, LODO, política
   FM-primeiro. (fonte: notebooks 08-09 + piloto LODO)
7. **Destilação distribucional** — desenho, resultados, Pareto 3 estratégias.
   (fonte: `docs/desenho-experimental-destilacao.md` + `results/distillation`)
8. **Conclusão** — síntese, limitações, trabalhos futuros.

**Tese mínima defensável** (se o Pilar A der resultado negativo): benchmark
14×18 rigoroso + framework validado + resultado negativo informativo de
destilação em regressão com análise de causa — ainda é uma dissertação
sólida. **Tese plena:** o acima com destilação funcionando + preprint.

## 4. Cronograma consolidado (registrado na prorrogação: conclusão nov, defesa dez)

| Mês | AI-1 | AI-2 | Dissertação |
|---|---|---|---|
| **jul/26** (agora) | Fechar run 14 modelos; refresh notebooks/deck | Harness destilação ✓; OOF real + piloto (gate) | — |
| **ago/26** | Entrega/apresentação [data a confirmar] | Grade completa destilação; LODO@14; ablações | Caps. 2-4 em rascunho |
| **set/26** | — | Extensão de datasets; Pareto 3-estratégias; relatório AI-2 | Caps. 5-6 em rascunho |
| **out/26** | — | **Preprint** + entrega AI-2 [data a confirmar] | Cap. 7 |
| **nov/26** | — | — | Consolidação, revisão, depósito |
| **dez/26** | — | — | **Defesa** |

## 5. Rastreabilidade — onde cada coisa vive (nada só em escopo local)

- **Código e resultados:** `github.com/Gvascons/master-thesis` (push a cada
  marco; resultados parciais commitados incrementalmente — política em vigor
  desde 14/07).
- **Decisões de pesquisa:** `docs/memo-*.md` (viabilidade dos 3 modelos,
  novidade da destilação) e `docs/desenho-experimental-destilacao.md`
  (pré-registro).
- **Metodologia:** `docs/metodologia-estatistica.md`.
- **Este programa:** `docs/programa-de-pesquisa.md` (revisar a cada mudança
  de rumo; a data de revisão no topo é o controle).
- **Tracking operacional:** lista de tarefas da sessão de trabalho (espelha
  as fases deste documento).
- **Documento da banca (prorrogação):** Desktop, `resumo-banca-temp.txt` —
  [pendente: conceitos das disciplinas + parecer/assinatura do orientador +
  envio ao sec-pos].

## 6. Registro de riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Pilar A com retenção ~0 nos pools completos | média | alto | Gate pré-registrado; resultado negativo é publicável; tese mínima independe |
| Concorrência publicar regressão antes de out | baixa-média | alto | Preprint cedo; pilar Pareto-3-estratégias segue nosso; monitorar arXiv mensalmente |
| TabFM (2 semanas de vida) mudar/receber paper que contradiga nossos achados | média | médio | Versão pinada; resultados são sobre a v1.0.1 e isso é declarado |
| GPU única limitar extensão de datasets | média | baixo | Custo dominante (OOF) é pago 1×; alunos são CPU |
| Prazos administrativos (formatos/datas das entregas) | ? | médio | [CONFIRMAR com secretaria — único item que não posso resolver] |
| Perda de trabalho local | — | — | Mitigado: push incremental em vigor |

## 7. Contrato de conduta científica (vinculante para todo o trabalho)

1. **Nenhuma afirmação de prioridade/novidade sem verificação online datada**
   (precedente: a checagem de 15/07 evitou reivindicar o que o Pocket FM já
   publicou).
2. **Pré-registro de hipóteses e critérios de decisão antes dos
   experimentos**; mudanças de desenho são documentadas com data e motivo.
3. **Resultados negativos são reportados** com a mesma proeminência.
4. **Distinção explícita** entre smoke test (qualitativo), piloto (decisório)
   e resultado (reportável).
5. **Todo número citado em texto tem origem rastreável** em artefato
   versionado (CSV/JSON/notebook).
6. **Correções são registradas, não reescritas** (precedente: retificação do
   AUC do tabkan em commit próprio).
