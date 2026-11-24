# Proveniência — sensibilidade aos 4 conjuntos CTR23 omitidos

Origem: auditoria externa independente de 23/11/2026 (banca simulada;
material integral em `~/Desktop/banca-simulada/outputs/`). O achado: a
regra de elegibilidade declarada da extensão (CTR23, 3.000–100.000 linhas,
sem vazamento, sem sobreposição com o núcleo) alcança 19 conjuntos; o
pipeline publicado rodou 15. Os quatro ausentes e o status de cada um:

| conjunto | OpenML | justificativa da omissão |
|---|---|---|
| `white_wine` | 44971 | subconjunto do `wine_quality` do núcleo (sobreposição real) |
| `california_housing` contínuo | 44977 | versão não binarizada do conjunto do núcleo (sobreposição real) |
| `superconductivity` | 44964 | à época (20/07) acreditava-se sobreposto ao núcleo; a errata de 19/10 (`docs/errata-datasets-core-regressao.md`) mostrou que a entrada "superconduct" sempre foi o `online_news` — a justificativa caiu e a omissão não foi revisitada |
| `naval_propulsion_plant` | 44969 | sem justificativa registrada |

Arquivos `xgboost_<ds>.json` (tuning Optuna 100 trials), `oof_tabpfn_<ds>.parquet`
(professor TabPFN OOF, K=5, semente 42), `resultados.csv` (todas as células) e
`openml_cache/` (bytes dos conjuntos) foram produzidos pela auditoria com o
protocolo de `scripts/distill.py`/`scripts/extension_ctr23.py` (um split,
`seed=42`, sementes de aluno {0,1,2}) e copiados daqui em 23/11/2026.

Verificação do autor: `scripts/ctr23_omitted_verify.py` re-treina o professor
de hold-out e os dois alunos quantílicos a partir dos caches e compara com
`resultados.csv` (→ `verify.csv`, `detalhe_retencao.csv`), e monta a tabela de
sensibilidade do placar (→ `results/distillation/ctr23_sensitivity.csv`).

Nota de recorte (23/11): o parecer da auditoria reporta "17/22 sem os
dois sobrepostos" — aritmética que corresponde a excluir white_wine +
superconductivity (as duas omissões com justificativa contemporânea).
O recorte adotado na dissertação é o conservador, por sobreposição
REAL com o núcleo (white_wine + california contínuo, este uma vitória
do destilado): 16/22, p = 0,026 (ctr23_sensitivity.csv).

Limite original: um único split por conjunto (como na auditoria). A
extensão de 5 splits FOI executada na madrugada de 24→25/11/2026
(runner `scripts/ctr23_omitted_multisplit.py`, splits 43–46; artefato
`multisplit.csv`); resultados integrados pela regra abaixo — todos os
splits reportados, agregado 17/24 (p=0,032) e, no recorte conservador,
16/22 (p=0,026); 18/24 apenas no split 45 (naval +0,022, único
cruzamento de zero).

## Regra de relato dos splits adicionais (fixada em 24/11/2026, ANTES de qualquer leitura dos splits 43–46)

O runner `scripts/ctr23_omitted_multisplit.py` executa os splits 43–46
(o 42 é o do adendo). Regra fixada antes dos resultados: serão
reportados TODOS os splits executados — retenção por (split, conjunto),
placar por split e agregado por conjunto pelo sinal da retenção média
entre splits — sem recorte posterior de splits ou conjuntos; execuções
que falharem serão reportadas como ausentes, nunca silenciosamente
excluídas; nenhum resultado parcial será citado como completo.
