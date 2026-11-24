# Adendo ao E2 — sensibilidade da normalização (24/11/2026)

**Estatuto.** Análise **pós-hoc**, feita depois de conhecidos os
resultados do E2 pré-registrado (`docs/desenho-melhorias-finais.md`,
seção E2, que permanece intocado). Motivação: o texto do capítulo 6
afirmava que normalizações alternativas não mudariam a leitura do
cruzamento, sem artefato que o medisse. Este adendo mede.

**Método.** `scripts/e2_norm_sensitivity.py`. Mesmas políticas fixas,
mesmo *regret* normalizado e mesmos dados
(`results/aggregated/test_results.csv` +
`results/latency/latency_adult.csv`) do E2 original; grade de
$w$ com passo 0,01. Três normalizações do score
$S = w \cdot n_{acc} + (1-w) \cdot n_{lat}$:

- **postos** (controle): postos normalizados [0,1] de acurácia e
  latência — reproduz o artefato original
  (`lodo_weighted_regret.csv`) com diferença zero nos $w$ comuns;
- **log**: acurácia em min–max [0,1] por conjunto (0 = melhor);
  latência em min–max de $\log_{10}$(µs/linha);
- **linear**: acurácia em min–max; latência em min–max dos µs/linha
  brutos.

Nas alternativas mudam, portanto, **as duas** normalizações (acurácia
e latência), não só a da latência.

**Resultado.** Artefatos versionados:
`results/aggregated/lodo_weighted_regret_norms.csv` (regret por
conjunto × normalização × política, grade de $w$ 0,05; md5
`621cd8980a1483122c72eb1490fe9557`) e
`results/aggregated/lodo_weighted_regret_norms_summary.csv`
(mediana/média agregadas, grade 0,01; md5
`df24ddf9571f54857ae93448af523b85`). O por-conjunto na grade 0,01 é
reproduzível executando o script (determinístico sobre os artefatos
de entrada). Recomputado de forma independente pela banca simulada
em 24/11/2026 — tabelas idênticas.
Último $w$ em que a política sempre-FM ainda é regret-ótima:

| normalização | estatística | grade 0,05 | grade 0,01 | peso na latência na inversão |
|---|---|---|---|---|
| postos | mediana | 0,85 | 0,83 | 15–20% |
| postos | média | 0,80 | 0,79 | 20–25% |
| log | mediana | 0,80 | 0,79 | 20–25% |
| log | média | 0,75 | 0,75 | 25–30% |
| linear | mediana | 0,70 | 0,69 | 30–35% |
| linear | média | 0,65 | 0,65 | 35–40% |

Na grade de 0,1 do estudo original, as faixas correspondentes são
10–20% (postos, mediana), 20–30% (log) e 30–40% (linear).

**Leitura.** As normalizações alternativas deslocam o ponto de
cruzamento na direção de **mais** tolerância à latência; a leitura
qualitativa que o arcabouço usa — existe um peso minoritário de
latência a partir do qual a recomendação deixa de ser o *foundation
model* — mantém-se nas três, e a normalização por postos, adotada no
desenho pré-registrado, é a que inverte mais cedo.
