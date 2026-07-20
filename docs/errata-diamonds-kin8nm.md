# ERRATA — o dataset rotulado "diamonds" sempre foi o kin8nm

> Descoberta em 20/07/2026, durante a análise consolidada da extensão CTR23.
> Retificação executada no mesmo dia (commit de retificação; histórico
> preservado). Este documento é a referência canônica do episódio.

## O erro

A entrada `diamonds` de `configs/datasets.yaml` apontava, desde a criação do
projeto, para `openml_id: 44980` — que é o **kin8nm** (cinemática de braço
robótico de 8 elos, 8.192 amostras, 8 features numéricas `theta1..theta8`),
e não o diamonds (44979; 53.940 amostras, preço de diamantes). O
`approx_samples: 54000` do config descrevia o dataset pretendido, não o
carregado — os dados reais sempre tiveram 8.192 linhas.

## Como foi descoberto

A extensão CTR23 incluía kin8nm (44980) por recomendação da verificação da
suíte. A análise consolidada acusou métricas **idênticas até a 4ª casa**
entre "kin8nm (extensão)" e "diamonds (core)" — probabilidade ~0 de
coincidência. A inspeção confirmou: mesmos `openml_id`, parquets
byte-idênticos, colunas `theta*`. Um indício anterior (19/07: "diamonds com
pool de 6.553, não ~43k") havia sido notado e incorretamente aceito como
aproximação de config — registro do lapso para o histórico.

## O que É e o que NÃO é afetado

- **A validade experimental NÃO é afetada.** kin8nm é um dataset legítimo da
  CTR23, executado sob protocolo íntegro em todos os experimentos. Apenas o
  RÓTULO estava errado.
- **Rótulos afetados e corrigidos:** resultados do benchmark (14 modelos ×
  1 dataset — arquivos raw renomeados e campo interno corrigido; agregados
  regenerados), artefatos da destilação (parquets OOF, distill/teacher_eval/
  ablation/pareto CSVs), config, e todos os documentos que citavam
  "diamonds" (deck/figuras regenerados na sequência).
- **Duplicata na extensão:** as 13 linhas "kin8nm" da extensão re-executaram
  o mesmo dado do core; foram movidas para
  `extension_kin8nm_repro_check.csv` — onde servem, involuntariamente, como
  **teste de reprodutibilidade ponta-a-ponta do pipeline de destilação**
  (números reproduzidos exatamente sob mesmos seeds/protocolo).
- **N da extensão:** recuperado com a inclusão do **diamonds verdadeiro**
  (`diamonds_real`, 44979) no pipeline da extensão — o dataset nunca havia
  sido executado.
- **Claim corrigida de passagem:** a verificação CTR23 apontara "diamonds já
  usado" como sobreposição — na verdade a sobreposição era com o kin8nm; o
  diamonds real era elegível o tempo todo.

## Lições registradas

1. `approx_samples` divergente do dado carregado é sinal de investigação
   obrigatória, não de tolerância (o lapso de 19/07).
2. Carregamento por ID pinado é necessário mas não suficiente — o ID
   precisa ser verificado contra a identidade pretendida na criação do
   config (uma checagem nome↔ID↔shape agora é trivial de adicionar ao
   download; pendência registrada).
3. A duplicação acidental foi o mecanismo de detecção — redundância entre
  fontes independentes tem valor de auditoria.
