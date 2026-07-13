# Memorando de viabilidade — KAN, TabKAN e TabFM (fase de ampliação do benchmark)

> Investigação conduzida em 13/07/2026 (três frentes de pesquisa independentes,
> fontes primárias verificadas online: arXiv, GitHub, PyPI, Hugging Face,
> OpenReview, Springer). Objetivo: decidir o caminho de integração de cada
> modelo ao pipeline (contrato `BaseModel` fit/predict/predict_proba, nested CV
> 5×3, Optuna TPE 25 trials DL, hold-out seed 42) no hardware do projeto
> (RTX 5080 16 GB — confirmada via nvidia-smi —, 32 GB RAM, Python 3.12).

## Veredito executivo

| Modelo | Viável? | Caminho | Esforço | Família | Tuning |
|---|---|---|---|---|---|
| KAN | Sim | efficient-kan (MIT) vendorizado + wrapper próprio | 2–4 dias | deep_learning | 25 trials |
| TabKAN | Sim | Reimplementação leve (camadas ChebyKAN + nosso loop) | 3–5 dias | deep_learning | 25 trials |
| TabFM | Sim | Pacote oficial Google, backend PyTorch, pesos do HF | 3–5 dias | foundation_model | zero-shot (0 trials) |

Nenhuma limitação de hardware bloqueante. Três decisões de política precisam
de acordo antes de codar (seção "Decisões em aberto").

---

## 1. KAN (Kolmogorov-Arnold Networks)

**Paper canônico:** Liu et al. (MIT/Caltech), arXiv 2404.19756, **Oral na ICLR
2025**, ≥1.885 citações. Arquitetura: inverte o MLP — funções de ativação
univariadas aprendíveis (B-splines) nas arestas, nós apenas somam. Viés
indutivo: aditividade suave de funções univariadas.

**Estado da evidência em tabular (ponto central para a dissertação):**
- A literatura é **contraditória e no agregado desfavorável**: Yu et al.
  (arXiv 2407.16674, comparação justa por parâmetros E FLOPs) — MLP vence KAN
  em 6 de 8 datasets de ML; KAN só domina em regressão simbólica.
- Poeta et al. (arXiv 2406.14529) — favorável em acurácia, mas sem baselines
  GBDT e com treino de só 10 épocas.
- **Nenhum benchmark tabular neutro de larga escala (TabArena, RealMLP) inclui
  KAN.** Toda alegação KAN ≥ GBDT vem dos proponentes. **Nosso benchmark
  preenche exatamente essa lacuna** — argumento de contribuição da tese.
- Custo esperado: ~2–4× o tempo do MLP (implementações eficientes); o "10×
  mais lento" do paper original refere-se ao pykan/LBFGS.

**Implementações:** todas as libs KAN congelaram em 2024–jan/2025. pykan
(oficial, MIT) é LBFGS-cêntrico, lento, 266 issues abertas — não serve como
motor de treino. **efficient-kan (MIT, ~4,6k stars)** reduz tudo a matmul,
é `nn.Module` puro → encaixa direto no nosso loop (Adam, batch 256, early
stopping, CE/MSE) e foi a base do benchmark de Poeta et al. (respaldo
citável). Alternativa se instável: FastKAN (Apache-2.0, RBFs + LayerNorm
embutido, ~3× mais rápido, mas se afasta da formulação B-spline canônica).

**Armadilha técnica nº 1 (mitigação obrigatória):** B-splines só existem
dentro do grid (default [-1,1]); com StandardScaler, valores caem fora e a
aresta degrada para o ramo SiLU. Solução no wrapper: **LayerNorm de entrada**
(ou grid_range=[-3,3] + update_grid de warm-up). Documentar na tese.

**Espaço Optuna (6 HPs, 25 trials):** lr 1e-4–1e-2 (log), largura 32–256
(log), profundidade 1–3, grid size {3,5,10,20}, weight decay 1e-6–1e-3 (log),
λ de regularização L1 {0}∪[1e-5,1e-2]. Fixar: k=3 (ordem da spline), SiLU,
AdamW. Sem dropout (não existe nas implementações; weight decay substitui).

**Hardware:** trivial — ~10⁵–10⁶ parâmetros, ativações << 1 GB. Riscos:
variância entre seeds maior que MLP (nested CV já mitiga); reportar contagem
de parâmetros honestamente (splines infladas por one-hot envenenam
comparações por paridade de parâmetros).

## 2. TabKAN

**Desambiguação (3 trabalhos com nomes próximos):**
- **(A) TabKAN** (Eslamian, Aghaei & Cheng, Univ. Kentucky) — arXiv
  2504.06559; peer-reviewed em *Machine Learning for Computational Science
  and Engineering* (Springer, nov/2025). **É o candidato.** Ressalvas: venue
  novíssima (vol. 1), ~1 citação, sem validação independente.
- (B) TabKANet (arXiv 2409.08806, Knowledge-Based Systems, ~8 citações) —
  KAN só como embedding numérico num Transformer. Alternativa de contingência
  (repo sem licença visível — verificar antes).
- (C) TKAN (arXiv 2405.07344) — séries temporais. Descartado.

**O que é:** framework modular de variantes KAN para tabular (Spline, Cheby,
Fast, Padé/Jacobi, Fourier, fKAN + KAN-Mixer), com one-hot para categóricos
(idêntico ao nosso pré-processamento DL) e módulo de interpretabilidade via
coeficientes. Melhor variante reportada: **ChebyKAN** (AUC média 0,857 vs
XGBoost 0,814 em 8 binários).

**Red flags dos resultados:** 6 dos 10 datasets < 8k amostras; XGBoost
possivelmente sub-tunado (AUC 0,726 no credit_g vs ~0,77+ que nós mesmos
obtivemos); assimetria de orçamento de tuning não clara; **sem regressão no
paper** (nosso benchmark produziria resultados de regressão inéditos para o
TabKAN); protocolo de treino atípico (L-BFGS full-batch, split único).

**Código:** repo oficial MIT + pacote `pip install tabkan` (~630 linhas,
inspecionado). **Incompatível com nosso protocolo**: API estilo pykan
(dicionário), L-BFGS full-batch, sem early stopping, sem predict_proba, sem
regressão. **Caminho: reimplementação leve** — camadas ChebyKAN (públicas,
MIT) + nosso loop de treino padrão, como fizemos com SAINT/STab. Desvio de
protocolo (Adam+batch vs L-BFGS) deve ser declarado na dissertação.

**Espaço Optuna (25 trials, variante fixa ChebyKAN):** profundidade 1–4,
largura 16–128 (log), grau do polinômio 2–5, lr 1e-4–1e-2 (log), weight
decay {0,1e-6,1e-5,1e-4}. (Paper usa 100 trials mas relata convergência em
15–20.)

**Hardware:** folga total (autores treinaram tudo, inclusive Covertype 581k,
numa A4500 de 20 GB).

## 3. TabFM (Google Research, 2026)

**Identificação:** github.com/google-research/tabfm — lançado em
**30/06/2026** (2 semanas atrás), por Kong & Das (Google Research). **Não há
paper arXiv até 13/07/2026** — citação hoje = blog oficial + model card do
Hugging Face + repo. Desambiguar na tese de "TabFMs" (arXiv 2310.07338,
LLM-based, 2023) e "TabularFM" (arXiv 2406.09837).

**Paradigma:** in-context learning zero-shot da **mesma família do TabPFN**
(prior-fitted, pré-treino em datasets sintéticos de SCMs), não um paradigma
distinto. Arquitetura híbrida: atenção de colunas + compressão de linha em 8
tokens + transformer ICL de 24 blocos (~1,5 bi de parâmetros, estimativa
nossa). **Enquadramento honesto na tese: replicação inter-laboratórios do
paradigma PFN/ICL (Google vs Prior Labs), não "segundo paradigma".**

**Suporte às tarefas:** binária sim; multiclasse **máx. 10 classes** →
`helena` (100 classes) fica de fora, exatamente como já ocorre com o TabPFN
(consistência mantida); regressão sim (checkpoint dedicado). Features ≤500
(nossos ≤90, OK).

**Viabilidade no hardware — confirmada:**
- Pesos liberados no HF (13,1 GB: 6,56 GB classificação + 6,59 GB regressão).
- Backend **PyTorch: 3–7 GB de VRAM** (medição independente em RTX 4090),
  ~40k linhas de contexto → cabe na RTX 5080 16 GB. **Evitar o backend JAX**
  (17–23 GB com pré-alocação XLA).
- Requer Python ≥3.11 (temos 3.12 ✓), torch==2.12.1 (verificar compatibilidade
  com o env atual — pode exigir env separado).
- Datasets grandes: contexto de 100k linhas é impraticável → usar
  `max_num_rows=50000`, espelhando o `tabpfn_max_samples: 50000` que já
  usamos (política idêntica = comparabilidade).
- Zero-shot → sem Optuna; só os folds do CV externo. Ensemble interno
  default n_estimators=32. **Piloto de latência obrigatório** antes da
  estimativa de custo total (sem benchmarks de latência publicados).

**Licença:** código Apache-2.0; pesos "TabFM Non-Commercial License v1.0" —
cobre explicitamente pesquisa acadêmica e benchmarking. Proibido redistribuir
pesos (linkar o HF, não versionar checkpoint).

**Resultados reportados:** TabArena (51 datasets), Google afirma zero-shot >
GBDTs tunados. Verificação independente parcial (10 datasets vs XGBoost
tunado) confirmou com ressalvas. Parquets brutos no repo merecem inspeção.

---

## Riscos científicos transversais (a registrar na dissertação)

1. **Maturidade assimétrica:** KAN é peer-reviewed de alto impacto (ICLR
   Oral) mas com evidência tabular desfavorável; TabKAN é peer-reviewed em
   venue fraca com ~1 citação; TabFM não tem paper. O benchmark deve
   enquadrá-los como "arquiteturas emergentes sob teste independente" — isso
   é contribuição, não fraqueza.
2. **Desvios de protocolo declarados:** TabKAN treinado com Adam+batch em vez
   de L-BFGS full-batch; KAN com LayerNorm de entrada; TabFM com subsample
   50k e exclusão do helena (idêntico ao TabPFN).
3. **Ineditismo:** resultados de regressão do TabKAN e comparação
   TabFM vs TabPFN sob protocolo neutro são, até onde verificamos, inéditos.
4. **Pinagem de versões:** TabFM tem 2 semanas de vida (pinar commit/versão);
   efficient-kan congelado (vendorizar o arquivo).

## Decisões em aberto (discutir antes de codar)

1. **TabKAN — variante única ou múltiplas?** Recomendação: fixar ChebyKAN
   (melhor do paper; múltiplas variantes explodiria o orçamento de 25 trials).
2. **TabFM — aceitar as políticas espelhadas do TabPFN?** (subsample 50k,
   helena fora, ensemble default da lib, citação via blog+model card).
3. **KAN — efficient-kan vendorizado com LayerNorm de entrada?** (alternativa:
   FastKAN se houver instabilidade).
4. **Correção à parte:** o doc da banca menciona RTX 5060; nvidia-smi
   confirma **RTX 5080**. Corrigir no txt do Desktop.

## Plano de execução proposto (após acordo)

1. KAN: vendorizar efficient-kan + wrapper + smoke tests (3 task types) +
   espaço Optuna + dry-run. [2–4 dias]
2. TabKAN: camadas ChebyKAN + wrapper + cabeça de regressão + sanity check
   vs paper (adult, credit_g) + espaço Optuna. [3–5 dias]
3. TabFM: env/deps + download dos pesos + wrapper espelhando TabPFN + piloto
   de latência (3 datasets) → decisão de custo → execução. [3–5 dias]
4. Benchmark completo dos 3 × 18 datasets sob protocolo idêntico (runner
   resumível existente). [tempo de máquina, estimar após pilotos]
5. Reagregação + notebooks + apresentação com 14 modelos.
