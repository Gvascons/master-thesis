# Tabela mestra de resultados — benchmark tabular 2026

> **14 modelos × 18 datasets** (10 binárias · 3 multiclasse · 5 regressão), resultado no hold-out (seed 42). Gerado por `scripts/build_results_table.py` a partir de `results/aggregated/test_results.csv` (+ latência e curvas RQ2). Cobertura: **250/252** — as 2 ausências são exclusões estruturais no helena (100 classes excede o limite de TabPFN e TabFM). Latência medida para os 14 modelos (adult); curvas RQ2 cobrem os 6 modelos varridos.

Métrica primária por tarefa: **binária → ROC-AUC (↑)**, **multiclasse → log-loss (↓)**, **regressão → RMSE (↓)**. Em cada coluna o **melhor valor está em negrito**; `Rank méd.` é o rank médio do modelo naquela tarefa (1 = melhor).


---

## 1. Scorecard — visão de relance (1 linha por modelo)

Ranks médios por tarefa (↓ melhor) + custo típico. `Rank n≤4k` é o rank no regime de poucos dados (RQ2, só os 6 modelos varridos).

| Modelo | Família | Impl. | Rank bin. | Rank multi | Rank reg. | Treino (s) | Tuning (s) | Latência (µs/linha) | Rank n≤4k |
|---|---|---|---|---|---|---|---|---|---|
| **TabFM** | Foundation | oficial | 2.7 | 1.0 | 1.2 | 0.2 | 0 | 43262.0 |  |
| **TabPFN** | Foundation | oficial | 4.2 | 2.5 | 3.0 | 1.3 | 0 | 7416.1 | 1.0 |
| **XGBoost** | GBDT | oficial | 5.0 | 6.0 | 5.0 | 0.7 | 132 | 0.3 | 2.8 |
| **CatBoost** | GBDT | oficial | 5.3 | 7.0 | 5.8 | 3.8 | 648 | 1.4 | 3.6 |
| **LightGBM** | GBDT | oficial | 6.3 | 8.0 | 5.6 | 0.7 | 106 | 1.9 |  |
| **TabM** | DL (MLP) | oficial | 6.6 | 3.7 | 8.4 | 33.8 | 3515 | 3.5 | 3.8 |
| **FT-Transformer** | DL (attn) | própria | 7.4 | 6.3 | 8.8 | 87.2 | 4761 | 28.9 | 4.4 |
| **SAINT** | DL (attn) | própria | 7.5 | 10.0 | 8.4 | 202.2 | 12245 | 44.7 |  |
| **RealMLP** | DL (MLP) | oficial | 7.7 | 4.3 | 5.2 | 36.9 | 1475 | 1.5 | 5.4 |
| **STab** | DL (attn) | própria | 8.8 | 3.7 | 9.6 | 198.9 | 14956 | 8644.8 |  |
| **MLP** | DL (MLP) | própria | 9.3 | 7.7 | 11.0 | 13.0 | 1150 | 2.4 |  |
| **KAN** | DL (KAN) | vendorizado | 11.0 | 12.0 | 9.8 | 11.9 | 750 | 3.1 |  |
| **TabKAN** | DL (KAN) | própria | 11.4 | 13.3 | 11.8 | 9.1 | 748 | 2.3 |  |
| **TabNet** | DL (attn) | comunidade | 11.8 | 11.7 | 11.4 | 65.0 | 6164 | 17.1 |  |

*Leitura rápida: **TabFM** lidera as três tarefas (zero-shot), com latência proibitiva (~100 ms/linha em contextos grandes); **TabPFN** é o vice consistente; os **GBDTs** são os all-rounders baratos; **KAN/TabKAN** ficam no fundo junto ao TabNet (teste independente desfavorável); o DL pesado custa minutos de treino sem liderar.*

---

## 2. Binária — ROC-AUC ↑ (10 datasets)

| Modelo | adult | amazon_employee | bank_marketing | cardiovascular_disease | credit_g | give_me_some_credit | higgs | magictelescope | phoneme | telco_customer_churn | **Rank méd.** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TabFM | **0.9322** | 0.8604 | **0.9442** | **0.8023** | 0.7763 | **0.8678** | **0.8403** | **0.9566** | **0.9717** | 0.8233 | 2.70 |
| TabPFN | 0.9204 | 0.8317 | 0.9397 | 0.7997 | 0.7746 | 0.8638 | 0.8111 | 0.9452 | 0.9529 | **0.8490** | 4.20 |
| XGBoost | 0.9301 | 0.8401 | 0.9339 | 0.8002 | 0.7667 | 0.8640 | 0.8117 | 0.9297 | 0.9429 | 0.8465 | 5.00 |
| CatBoost | 0.9297 | **0.8869** | 0.9333 | 0.8001 | 0.7720 | 0.8646 | 0.8107 | 0.9314 | 0.9454 | 0.8431 | 5.30 |
| LightGBM | 0.9299 | 0.8435 | 0.9292 | 0.8001 | 0.7656 | 0.8640 | 0.8111 | 0.9284 | 0.9416 | 0.8446 | 6.30 |
| TabM | 0.9161 | 0.8108 | 0.9369 | 0.7991 | 0.7837 | 0.8268 | 0.8193 | 0.9328 | 0.9388 | 0.8425 | 6.60 |
| FT-Transformer | 0.9165 | 0.7835 | 0.9357 | 0.7993 | **0.7873** | 0.8270 | 0.8096 | 0.9278 | 0.9362 | 0.8465 | 7.40 |
| SAINT | 0.9201 | 0.7793 | 0.9326 | 0.7998 | 0.7651 | 0.8273 | 0.8119 | 0.9338 | 0.9358 | 0.8460 | 7.50 |
| RealMLP | 0.9190 | 0.8068 | 0.9303 | 0.7980 | 0.7362 | 0.8625 | 0.8140 | 0.9366 | 0.9482 | 0.8417 | 7.70 |
| STab | 0.9172 | 0.6769 | 0.9344 | 0.7991 | 0.7695 | 0.8274 | 0.8184 | 0.9278 | 0.9138 | 0.8413 | 8.80 |
| MLP | 0.9128 | 0.8208 | 0.9294 | 0.7981 | 0.7744 | 0.8212 | 0.8133 | 0.9291 | 0.9213 | 0.8391 | 9.30 |
| KAN | 0.9151 | 0.8171 | 0.9274 | 0.7979 | 0.7494 | 0.8190 | 0.8018 | 0.9165 | 0.8952 | 0.8456 | 11.00 |
| TabKAN | 0.9140 | 0.8125 | 0.9252 | 0.7955 | 0.7770 | 0.8187 | 0.7908 | 0.9139 | 0.8939 | 0.8421 | 11.40 |
| TabNet | 0.9121 | 0.7888 | 0.9268 | 0.7805 | 0.7462 | 0.8175 | 0.8059 | 0.9352 | 0.9232 | 0.8314 | 11.80 |

---

## 3. Multiclasse — log-loss ↓ (3 datasets)

*TabPFN não roda o helena (100 classes > limite); rank médio sobre os 2 datasets restantes.*

| Modelo | covertype | helena | jannis | **Rank méd.** |
|---|---|---|---|---|
| TabFM | **0.0966** |  | **0.5709** | 1.00 |
| TabPFN | 0.1644 |  | 0.6583 | 2.50 |
| STab | 0.1786 | 2.5342 | 0.6698 | 3.67 |
| TabM | 0.2067 | **2.5097** | 0.6572 | 3.67 |
| RealMLP | 0.1894 | 2.5144 | 0.6817 | 4.33 |
| XGBoost | 0.1934 | 2.6010 | 0.6801 | 6.00 |
| FT-Transformer | 0.2076 | 2.5420 | 0.6774 | 6.33 |
| CatBoost | 0.2035 | 2.5800 | 0.6836 | 7.00 |
| MLP | 0.2118 | 2.5260 | 0.6874 | 7.67 |
| LightGBM | 0.2009 | 2.6265 | 0.6836 | 8.00 |
| SAINT | 0.2289 | 2.6090 | 0.6887 | 10.00 |
| TabNet | 0.2423 | 2.6758 | 0.7027 | 11.67 |
| KAN | 0.2496 | 2.6376 | 0.7080 | 12.00 |
| TabKAN | 0.2577 | 2.7187 | 0.7257 | 13.33 |

---

## 4. Regressão — RMSE ↓ (5 datasets)

*RMSE está na escala de cada alvo — compare **dentro** da coluna, não entre colunas.*

| Modelo | california_housing | diamonds | superconduct | wine_quality | year_prediction | **Rank méd.** |
|---|---|---|---|---|---|---|
| TabFM | **0.2346** | 0.0652 | **0.8269** | **0.5747** | **8.3634** | 1.20 |
| TabPFN | 0.2570 | 0.0667 | 0.8284 | 0.6129 | 8.5952 | 3.00 |
| XGBoost | 0.2637 | 0.1117 | 0.8308 | 0.5975 | 8.6801 | 5.00 |
| RealMLP | 0.2801 | **0.0630** | 0.8401 | 0.6728 | 8.6269 | 5.20 |
| LightGBM | 0.2641 | 0.1075 | 0.8320 | 0.6131 | 8.6727 | 5.60 |
| CatBoost | 0.2653 | 0.0859 | 0.8331 | 0.6005 | 8.6949 | 5.80 |
| TabM | 0.2798 | 0.0656 | 0.8482 | 0.6607 | 8.9224 | 8.40 |
| SAINT | 0.2873 | 0.0653 | 0.8464 | 0.6869 | 8.8207 | 8.40 |
| FT-Transformer | 0.2851 | 0.0677 | 0.8432 | 0.6774 | 8.8727 | 8.80 |
| STab | 0.2905 | 0.0820 | 0.8415 | 0.7077 | 8.8357 | 9.60 |
| KAN | 0.3371 | 0.1513 | 0.8457 | 0.6597 | 8.7864 | 9.80 |
| MLP | 0.3002 | 0.0821 | 0.8523 | 0.6629 | 9.3563 | 11.00 |
| TabNet | 0.3326 | 0.0808 | 0.8594 | 0.6993 | 8.8622 | 11.40 |
| TabKAN | 0.3324 | 0.1526 | 0.8460 | 0.6941 | 8.8662 | 11.80 |

---

## 5. Custo — treino, tuning e latência

Medianas entre datasets; latência medida no `adult`. **~300× de variação no treino, ~26.000× na latência** — e o ranking de custo é quase o inverso da intuição.

| Modelo | Família | Treino final (s) | Tuning total (s) | Latência (µs/linha) |
|---|---|---|---|---|
| XGBoost | GBDT | 0.69 | 132 | 0.33 |
| CatBoost | GBDT | 3.76 | 648 | 1.43 |
| RealMLP | DL (MLP) | 36.90 | 1475 | 1.52 |
| LightGBM | GBDT | 0.66 | 106 | 1.91 |
| TabKAN | DL (KAN) | 9.15 | 748 | 2.32 |
| MLP | DL (MLP) | 12.95 | 1150 | 2.37 |
| KAN | DL (KAN) | 11.94 | 750 | 3.07 |
| TabM | DL (MLP) | 33.76 | 3515 | 3.48 |
| TabNet | DL (attn) | 64.97 | 6164 | 17.09 |
| FT-Transformer | DL (attn) | 87.22 | 4761 | 28.87 |
| SAINT | DL (attn) | 202.19 | 12245 | 44.69 |
| TabPFN | Foundation | 1.35 | 0 | 7416.10 |
| STab | DL (attn) | 198.85 | 14956 | 8644.82 |
| TabFM | Foundation | 0.18 | 0 | 43262.04 |

*A inversão do TabPFN: ~1s de treino e **zero tuning**, mas ~7.400 µs/linha de inferência (~22.000× o XGBoost). O tier rápido de latência inclui os 3 GBDTs **e** os DL tipo-MLP (RealMLP/MLP/TabM); só atenção, in-context e sampling são lentos.*

---

## 6. RQ2 — curvas de aprendizado / crossover DL↔GBDT

Onde a liderança muda de mãos conforme o pool cresce (envelope best-of-família, 500 → pool completo, 3 sementes). Ver figura `learning_curves.png` e o notebook `10_learning_curves.ipynb`.

| Dataset | Tarefa | Vence com pouco dado | Vence com muito dado | GBDT reassume em n≈ |
|---|---|---|---|---|
| give_me_some_credit | binary | GBDT | GBDT | 500 |
| higgs | binary | GBDT | DL | nunca (DL à frente) |
| adult | binary | DL | GBDT | 4000 |
| jannis | multiclass | GBDT | DL | nunca (DL à frente) |
| year_prediction | regression | GBDT | DL | nunca (DL à frente) |

**Rank no regime de poucos dados (n ≤ 4000), 1 = melhor:**

| Modelo | Rank médio n≤4k |
|---|---|
| TabPFN | 1.00 |
| XGBoost | 2.80 |
| CatBoost | 3.60 |
| TabM | 3.80 |
| FT-Transformer | 4.40 |
| RealMLP | 5.40 |

*Em 3 de 5 datasets o DL ultrapassa os GBDTs e não devolve a liderança; o TabPFN é o melhor em **todos** os datasets no regime de poucos dados.*

---

*Para regenerar: `uv run python scripts/build_results_table.py`.*

