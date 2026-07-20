> **ERRATA 20/07/2026:** o dataset historicamente rotulado "diamonds" e o kin8nm (OpenML 44980) — ver `docs/errata-diamonds-kin8nm.md`. Rotulos deste documento ja corrigidos; o diamonds real (44979) entrou pela extensao como `diamonds_real`.

# Preprint outline — working title:
# "Distilling Tabular Foundation Models Beyond Classification:
#  Distributional Regression and the Distill/Compress/Cache Pareto Frontier"

> Status: skeleton v1 (20/07/2026). Language: English. Target: arXiv preprint
> by Oct/2026; venue candidates to shortlist after results freeze (workshop
> track of a major ML conf, or a strong applied venue — decide with advisor).
> Every number below is traceable to a versioned artifact (paths noted).
> Authors: J. G. A. Vasconcelos, G. C. Vasconcelos [+ advisor decision].

## Abstract (draft skeleton, ~180 words)

- Tabular foundation models (TabPFN v2.5, TabFM) now lead accuracy across
  tasks, but in-context inference costs 4-5 orders of magnitude more latency
  than GBDTs (measured: 7.4ms-43ms/row vs 0.33us/row).
- Prior distillation work covers classification (Pocket FM); we study
  **regression**, where the teacher's output is a predictive DISTRIBUTION.
- Contributions: (1) distributional distillation via quantile-curve
  transfer — students retain 13-64% of the teacher's CRPS advantage at
  microsecond latency; (2) two robust negative results — point-distillation
  gains vanish at realistic pool sizes regardless of teacher strength;
  (3) OOF labeling matters for CALIBRATION, not accuracy, in regression;
  (4) the first three-strategy Pareto frontier (distill vs context
  compression vs ensemble reduction); (5) conditions-of-value: when the
  student's native distributional objective is strong, no teacher is needed.

## 1. Introduction
- The 2026 accuracy/latency inversion (source: benchmark @14,
  `notebooks/TABELA_RESULTADOS.md`; latency `results/latency/latency_adult.csv`).
- The validated FM-first policy and its single obstacle
  (`results/aggregated/lodo_validation_14.csv`).
- Contributions list (above). Honest scope: 5 datasets + planned CTR23
  extension; single GPU.

## 2. Related work
- Source: `dissertation/cap3-related-work-outline.md` §3.5 + novelty memo
  (`docs/memo-novidade-destilacao.md`, dated 15/07/2026).
- Pocket FM (must-cite, methodological base for OOF), Prior Labs engine,
  TabDistill, GAM distillation, TACO/MotherNet (acceleration alternatives),
  Bucila/Caruana lineage, quantile regression + CRPS foundations.

## 3. Method
- Teachers: TabPFN v2.5 (native quantile grid, 19 levels), TabFM (point).
- Students: XGBoost point / XGBoost multi-quantile (native pinball for
  hard; multi-output MSE on teacher quantile curves for soft — crossing
  fixed by sorting), MLP-quantile (pinball / MSE symmetric pair).
- OOF labeling (K=5); capacity control (students use the benchmark's tuned
  HPs — only the target changes); metrics: RMSE, CRPS (pinball integral),
  PICP80/90 + widths, normalized gap retention.
- Pre-registration: hypotheses/gates in
  `docs/desenho-experimental-destilacao.md` (+ dated addendum).

## 4. Experimental setup
- 5 OpenML regression datasets (5k-80k pools), same fixed hold-out as the
  parent benchmark; 3 seeds; RTX 5080 16GB (VRAM constraints documented —
  memory-saving mode finding, §6).
- [EXTENSION SLOT: +N CTR23 datasets — pending suite verification.]

## 5. Results
- 5.1 Point distillation fails at scale (H1): both teachers; the
  smoke-vs-pilot size contrast + pool sweep (wine replicates at n=800,
  gone by 2k; california noisy) — reported as observation.
  (`results/distillation/distill.csv`, `distill_cap*.csv`)
- 5.2 Distributional distillation, N=15 (core + CTR23 extension):
  positive CRPS-gap retention in 11/15 (median +0.13, max +0.64; sign
  test p=0.059). Failure modes characterized: no-teacher-edge (wine) and
  heavy-tailed/price-scale targets (fifa, diamonds, health_insurance) —
  log-target moderator queued as ablation. Calibrated claim, not a
  universal one. (`extension.csv` + `distill.csv`)
- 5.3 The Pareto frontier (H3): figure
  (`results/figures/pareto_distill.pdf`); below ~250us (california) /
  ~1.3ms (year) only students exist; distilled = CRPS-optimal. TabFM
  anchor at 394ms/row. Side-findings: 1-member ensemble ~= 8 at 1/7
  latency; the 25k->50k context step costs 3.3x latency for 0.05 RMSE.
  (`results/distillation/pareto.csv`)
- 5.4 Why OOF: in-sample labels do not hurt RMSE/CRPS in regression —
  they erode CALIBRATION (PICP80 -4 to -10 p.p.); fixed-context cell
  isolates leakage from context size. (`ablation_insample.csv` + ctxfix)
- 5.5 Student family matters: hard-label quantile MLP matches distilled
  XGB with no teacher on small/numeric data, blows up on year where
  distillation rescues it (9.66->4.91 CRPS). Conditions-of-value table.
  (`mlp_students.csv`)

## 6. Discussion
- What distillation transfers (distribution shape, calibration — not the
  mean); when to distill (weak/unstable native student + teacher with a
  verified OOF edge); practical recipe box.
- Hardware notes as findings (16GB-class GPUs and the full-context teacher).

## 7. Limitations
- N datasets; single student-HP policy (benchmark-tuned, no re-tuning per
  target regime); 19-quantile CRPS approximation; TabFM distributional
  output unexplored (bins — exploratory item); TabFM is 3 weeks old,
  version-pinned, no peer-reviewed reference.

## 8. Reproducibility statement
- Public repo, pinned versions, pre-registered design with dated addenda
  and a public correction (commit 64b410a) — the conduct contract as a
  feature.

## TODO before submission
- [x] CTR23 extension executed (10 datasets; erratum episode documented)
- [ ] Log-target ablation on the 4 failure datasets
- [ ] Fixed-context ablation folded into §5.4  [running]
- [ ] Figure polish (fonts/sizes for two-column), table formatting
- [ ] Related-work re-verification sweep (arXiv monitor: "tabular
      foundation model distillation", monthly until submission)
- [ ] Advisor review pass; author list/order decision
- [ ] Venue shortlist + formatting
