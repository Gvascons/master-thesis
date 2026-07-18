# Roteiro da apresentação — benchmark tabular 2026 (14 modelos)

> **Como usar:** roteiro estruturado seguindo o `00_presentation.ipynb` de
> cima a baixo. **[FIGURA: nome.png]** = mostrar a imagem (mesmos `show()` do
> deck). Números em **negrito** conferidos contra os artefatos em 18/07/2026.
> Para a versão corrida/informal, ver `FALA_APRESENTACAO.md`.

---

## Abertura — a tese em dois movimentos

"'Em dados tabulares, GBDT sempre vence' — a frase foi formada antes da geração
2024-2026. Re-testamos com rigor: **14 modelos × 18 datasets**, nested CV,
tuning bayesiano, testes frequentistas e bayesianos. E a resposta tem dois
movimentos: **(1)** entre GBDTs e deep learning a acurácia **empata** — e a
decisão migra pra custo, latência, robustez e estrutura; **(2)** a geração 2026
de foundation models **quebra o empate** — ao custo de ~5 ordens de magnitude
em latência. O arcabouço multicritério ficou *mais* necessário, não menos."

## §1 — Os 14 modelos (honestidade de proveniência)

7 de bibliotecas oficiais (3 GBDTs, TabPFN, TabFM, RealMLP, TabM) · 1 da
comunidade (TabNet) · 1 vendorizado com adaptações (KAN/efficient-kan) ·
**5 implementações próprias** (MLP, FT-Transformer, SAINT, STab, TabKAN).
Reimplementações com *sanity check* contra os papers (TabKAN: 0,91 vs ~0,90
reportado). TabFM: lançado 30/06/2026, **sem paper peer-reviewed** — citamos
blog + model card + versão pinada.

## §2-3 — Protocolo e datasets
Nested CV 5×3 · Optuna TPE (GBDT 100 / DL 25 trials) · hold-out 20% intocado ·
estratificação em tudo · 18 datasets (10 bin, 3 multi, 5 reg), 1K-581K, cap
100K. **[FIGURAS: dataset_landscape.png, class_imbalance.png]**

## ATO I — O Empate (geração ≤2025)

**[FIGURAS: heatmap_classification.png, average_ranks.png, cd_diagram_binary.png]**

- Friedman rejeita (χ²=31,3, p=5×10⁻⁴; **Iman-Davenport** F=4,1, p=1×10⁻⁴),
  post-hoc não separa o topo.
- **Bayesiano (ROPE ±0,01 AUC), escopo declarado nos 11 originais: 8/10
  praticamente equivalentes ao TabPFN.**
- **Sensibilidade do ROPE: 0,005/0,01/0,02 → 2/8/10** — no limiar estrito
  sobrevivem XGBoost (0,65) e CatBoost (0,54). A conclusão aguenta stress.
- **η²=0,20 → 80% da variância é intra-família** — o modelo importa mais que
  a família. TabNet consistentemente o pior (até as KANs chegarem).

## ATO II — Custo

**[FIGURAS: training_time.png, pareto_binary.png, inference_time.png]**
Treino ~300× de variação; latência ~26.000× (nos 11 originais). Fronteira de
Pareto: {TabPFN, XGBoost, LightGBM}. A inversão do TabPFN (barato de treinar,
caro de servir) — que o TabFM leva ao extremo (§6.2).

## ATO III — Reversões

**[FIGURAS: cd_diagram_multiclass.png, imbalance_robustness.png,
feature_type_sensitivity.png, robustness_riskmap.png]**
Multiclasse revertia pro DL (N=3, descritivo); GBDT mais forte sob
desbalanceamento; one-hot quebra atenção no extremo categórico; TabPFN com o
melhor piso de risco.

## §6.1 — Curvas de aprendizado (RQ2)

**[FIGURA: learning_curves.png]** 6 modelos × 5 datasets × 3 sementes,
500→pool. Crossover é específico do problema (3/5 DL ultrapassa e não devolve;
adult inverte em n≈4000). **TabPFN rank 1,0 em TODOS os datasets em n≤4k.**

## §6.2 — A QUEBRA DO EMPATE (geração 2026) ⭐ NOVO

| | binária | multiclasse | regressão |
|---|---|---|---|
| **TabFM** | **2,7** | **1,00** | **1,2** |
| TabPFN | 4,2 | 2,50 | 3,0 |
| melhor clássico | xgboost 5,0 | stab/tabm 3,67 | xgboost 5,0 |
| kan / tabkan | 11,0 / 11,4 | 12,0 / 13,3 | 9,8 / 11,8 |

- **TabFM: nº 1 absoluto em 13/18 datasets** (participa de 17; helena excede
  o limite de 10 classes — mesma exclusão do TabPFN). Zero-shot, fit de
  segundos, latência ~100 ms/linha (~300.000× o XGBoost, medição de piloto).
- **KANs: teste independente desfavorável** — último terço nas 3 tarefas;
  contradiz as alegações do paper do TabKAN (baseline sub-tunado é a
  explicação provável); confirma a literatura cética (Yu et al. 2024).
  Primeiro teste neutro da família, até onde verificamos.
- **LODO do framework:** árvore de meta-features NÃO generaliza (hit 0,11 vs
  baseline 0,44); política **"FM primeiro, desvie por restrição"** tem regret
  **mediano 0,000** @14 (era 0,018 @11). O flowchart (constraint-driven) sai
  validado.
- Cautela estatística dita no deck: com k=14 e N=10 o CD é largo — a
  separação do TabFM vem da **consistência (13/18) + bayesiano**, não do
  Nemenyi.

## §7-8 — Framework (Ato IV)

**[FIGURAS: metafeature_correlations.png, winner_decision_tree.png (agora
"ilustração descritiva", validação LODO citada), decision_matrix.png,
decision_flowchart.png]**
Matriz multicritério + flowchart de 3 perguntas — validados: as perguntas
certas eram as restrições.

## §9 — Interpretabilidade
**[FIGURAS: feature_importance_comparison.png, shap_beeswarm_adult.png,
prediction_agreement.png]** GBDT interpretável por construção; FMs caixa-preta
— mais um eixo da matriz.

## §10 — Honestidade + AI-2

Caveats: N pequeno (poder), piso do Wilcoxon em N=5 (0,0625 — impossível
rejeitar), helena 2× excluído, TabFM v1.0.1 sem paper, multiclasse descritivo.
**AI-2 = destilação distribucional de FMs para regressão** (novidade checada e
datada: classificação já coberta pelo Pocket FM mai/2026 — nosso recorte é
regressão + Pareto destilar/comprimir/cachear + TabFM teacher). Desenho
pré-registrado; smoke com os sinais previstos (aluno herda calibração: PICP80
0,73 vs 0,36 do controle); piloto em execução; meta: preprint out/2026.

---

## Colinha de números (@14, conferidos 18/07/2026)

| Número | Valor |
|---|---|
| Modelos × datasets · células | 14 × 18 · **250/252** |
| Equivalentes ao TabPFN (ROPE 0,01, geração ≤2025) | 8/10 |
| Sensibilidade ROPE 0,005/0,01/0,02 | 2/8/10 |
| Variância intra-família | 80% (η²=0,20) |
| TabFM: vitórias absolutas | **13/18** |
| TabFM: rank médio bin/multi/reg | 2,7 / 1,00 / 1,2 |
| TabFM: latência (piloto, contexto 35-50k) | ~100 ms/linha (~300.000× XGBoost) |
| KAN / TabKAN rank binária | 11,0 / 11,4 (TabNet: 11,8) |
| LODO: hit da árvore vs baseline | 0,11 vs 0,44 |
| LODO@14: regret mediano FM-primeiro | **0,000** |
| Wilcoxon N=5: piso do p-valor | 0,0625 (Holm satura em 1,0) |
| Smoke destilação: PICP80 soft vs hard | 0,73 vs 0,36 (nominal 0,80) |
