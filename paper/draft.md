# Distilling Tabular Foundation Models Beyond Classification: Distributional Regression and the Distill/Compress/Cache Pareto Frontier

> **Draft v1** (20/07/2026) — full prose, Markdown master; LaTeX conversion at
> venue-formatting time. Every number is traceable to a versioned artifact in
> the public repository (paths in comments throughout). Authors and order:
> pending advisor decision. Competition re-verification sweep: done 20/07/2026 — cut confirmed open (see docs/memo-novidade-destilacao.md).

## Abstract

Tabular foundation models now lead predictive accuracy: in our companion
benchmark of 14 models across 18 OpenML datasets, TabFM (Google, 2026) ranks
first in binary classification, multiclass, and regression, with TabPFN v2.5
a consistent second. This leadership carries an extreme cost: in-context
inference runs at 7.4–43 ms/row — four to five orders of magnitude slower
than gradient-boosted trees (0.33 µs/row). Prior work distills tabular
foundation models for *classification*; we study *regression*, where the
teacher's output is a predictive **distribution**. Across 15 datasets we
find: (1) **point** distillation fails at realistic pool sizes regardless of
teacher strength — a robust negative replicated with two teachers; (2)
**distributional** distillation — transferring the teacher's quantile curves
into a multi-quantile student — retains a median 19% and up to 100%+ of the
teacher's CRPS advantage (positive in 16/20 datasets, including one
born-again case where the student surpasses its teacher; one-sided sign
test p = 0.006, Wilcoxon p = 0.045, Bayesian P(ret>0) = 0.70) at
microsecond latency,
making the distilled student the only sub-millisecond distributional system
on our three-strategy Pareto frontier (distill vs. context compression vs.
ensemble reduction); (3) out-of-fold labeling matters for the student's
**calibration**, not its accuracy — a deconfounded refinement of prior
classification findings; and (4) failures are diagnosable, yielding a
practical decision rule: try target transforms and a strong native student
first, and distill when a genuine teacher edge persists — cheaply verifiable
via OOF labeling. All experiments are pre-registered, seed-controlled, and
publicly reproducible on a single 16 GB GPU.

## 1. Introduction

For a decade, the practical advice for tabular prediction was stable:
gradient-boosted decision trees (GBDTs) win. The 2024–2026 generation of
tabular deep learning and, decisively, of tabular foundation models (TFMs)
has overturned the accuracy half of that advice. In a uniform-protocol
benchmark we ran of 14 models × 18 OpenML datasets (nested cross-validation,
per-family preprocessing, equal tuning budgets), the zero-shot TabFM ranks
first on all three task types and is the single best model on 13 of 18
datasets; TabPFN v2.5 is the consistent runner-up.
<!-- notebooks/TABELA_RESULTADOS.md; results/aggregated/test_results.csv -->

The cost half of the advice, however, has inverted rather than disappeared.
In-context learners carry their training set as context at prediction time:
we measure 7,416 µs/row for TabPFN and 43,262 µs/row for TabFM on the adult
dataset, against 0.33 µs/row for tuned XGBoost — a factor of 22,000× and
131,000× respectively. <!-- results/latency/latency_adult.csv --> A
leave-one-dataset-out policy analysis over the same benchmark shows that
"use the foundation model by default" has median normalized regret 0.000 —
the *only* obstacle to that policy is serving cost.
<!-- results/aggregated/lodo_validation_14.csv -->

Knowledge distillation is the natural attack on that obstacle, and recent
work has established it for tabular **classification**. Regression, however,
is not classification with a continuous label: modern TFMs output a full
*predictive distribution* (TabPFN's bar-distribution head), and the value of
that distribution — calibrated uncertainty at prediction time — is precisely
what a point-label student cannot learn from raw data alone. This paper asks
what distillation can and cannot transfer in tabular regression, at what
serving cost, and under which verifiable conditions.

**Contributions.**
1. **A robust negative:** point distillation of TFMs fails at realistic pool
   sizes (positive in only 2/15 datasets), replicated with two teachers of
   very different strength, with a pool-size sweep showing the residual gain
   is a small-data phenomenon on some datasets only.
2. **A positive on the open axis:** distributional distillation via
   quantile-curve transfer is positive in 16/20 datasets — the complete
   eligible CTR23 pool under a rule fixed before any results — with median
   retention 19%, maximum above 100% (a born-again student), and agreement
   across instruments (sign test p = 0.006; Wilcoxon p = 0.045; Bayesian
   signed-rank P(retention>0) = 0.70 at ROPE ±0.05), at microsecond
   latency.
3. **The first three-strategy Pareto frontier** for TFM serving in
   regression — distillation vs. context compression vs. ensemble reduction
   — measured end-to-end on the same hold-outs; below ~250 µs/row (up to
   ~1.3 ms/row on our largest dataset) the distilled student is the only
   distributional option, and side-findings show ensemble reduction is
   underrated (1 member ≈ 8 at 1/7 the latency) while the last context
   doubling is grossly overpriced on 16 GB GPUs.
4. **A deconfounded account of why out-of-fold labeling matters in
   regression:** in-sample teacher labels do not hurt student accuracy or
   CRPS — they erode interval *coverage* (−0.5 to −5.1 p.p. at fixed context
   size), i.e., the teacher's in-context overconfidence transfers to the
   student.
5. **A practical decision rule** distilled (pun intended) from two refuted
   hypotheses: before distilling, apply cheap target transforms and try a
   strong native quantile student; distill when a genuine teacher edge
   survives both — verifiable with one inexpensive OOF pass.

## 2. Related Work

**Tabular foundation models.** TabPFN v2/v2.5 established prior-fitted
in-context learning for tabular data; TabICL, TabFlex and MotherNet explore
faster or amortized variants; TabFM (Google, 2026) is an independent
implementation of the paradigm with a hybrid column-attention/row-compression
architecture, released with weights but — as of this writing — no
peer-reviewed reference (we cite the official blog and model card, and pin
the version). Our companion benchmark provides the neutral evidence of the
accuracy leadership and the latency inversion that motivate this paper.

**Distilling TFMs.** Pocket FM distills classification TFMs (TabICLv2,
TabPFNv2.6, LimiX, Orion-MSP) into XGBoost/CatBoost/MLP students on 153
datasets, retaining 96.5% of teacher AUC at 38–860× speedups, and
establishes stratified out-of-fold labeling as necessary; a companion paper
extends the analysis to clinical data. Prior Labs ships a closed-source
distillation engine for TabPFN-2.5. TabDistill targets few-shot MLP
students; another line distills TFMs into GAMs for interpretability. None of
these addresses regression, distributional outputs, or a cross-strategy
serving frontier — the axes of this paper. The classical lineage (model
compression via soft labels; born-again trees) supplies the mechanics we
adapt to quantile curves.

**Acceleration without distillation.** TACO compresses the in-context set to
~1% with large speedups; CRUMB selects a compact context via MMD for
efficient PFN inference without retraining — a representative method of the
context-compression arm our frontier measures; MotherNet amortizes fitting
into a hypernetwork that emits MLP weights; TL-ANDI combines locally
distilled labels with optimal-transport context selection for cross-task
transfer (orthogonal goal, no fast students, no distributional axis); simple
engineering (context truncation, ensemble reduction, KV-caching) is
folklore. Our frontier measures the engineering strategies head-to-head with
distillation on identical hold-outs. On OOF labeling specifically, the
health-data companion of Pocket FM already establishes stratified OOF for
classification soft-labels; our contribution on that axis is confined to
what OOF protects in *distributional regression* — calibration, not
accuracy (§5.4).

**Distributional regression.** Quantile regression with pinball loss, CRPS
as a proper scoring rule, and coverage/sharpness diagnostics are standard;
XGBoost ≥2.0 supports native multi-quantile objectives, which we use both as
a hard-label control and as the student body for curve transfer.

## 3. Method

**Teachers.** TabPFN v2.5 is our distributional teacher (v2.5 rather than
the newer TabPFN-3 for version-pinned comparability with the companion
benchmark and verified 16 GB-GPU operation; nothing in our method is
version-specific): its native API
returns arbitrary quantiles of the predictive distribution, from which we
take a 19-level grid (τ = 0.05, …, 0.95). TabFM is a point-only teacher —
we verify architecturally (its regression head decodes a single scalar) and
empirically (its 8-member ensemble spread yields CRPS 0.524 vs. the
TabPFN teacher's 0.069 on california_housing, with 16.3% coverage at 80%
nominal) that no usable predictive distribution is available from it. Both
teachers run with an 8-member inference ensemble and a 50k-row context cap,
matching the companion benchmark's policies (for TabFM, 8 members measured
accuracy-equivalent to the library default of 32 at ¼ the cost).

**Out-of-fold labeling.** Teacher targets for every training row come from a
K=5 out-of-fold scheme: the teacher conditioned on 4/5 of the pool labels
the held fifth. Labels are cached once per (dataset, teacher) and reused by
all student variants.

**Students and targets.** All students inherit the tuned hyperparameters of
the corresponding baseline from the companion benchmark — capacity is held
fixed and only the target changes, isolating the distillation effect.
- *XGBoost point*, trained on true labels (control), on the teacher's OOF
  mean (soft), or on a 0.8/0.2 mixture.
- *XGBoost multi-quantile*: hard variant uses the native pinball objective on
  true labels; soft variant fits the 19 teacher quantile curves as a
  multi-output regression (crossing repaired by row-wise sorting).
- *MLP-quantile*: the benchmark's MLP body with a 19-output head; pinball on
  labels (hard) or MSE on teacher curves (soft) — symmetric with the XGBoost
  pair, to test whether the student family matters.

**Metrics.** RMSE and MAE for point quality; CRPS via the pinball-loss
integral over the grid; PICP80/90 with interval widths for calibration; and
a normalized retention score, (CRPS_hard − CRPS_soft)/(CRPS_hard −
CRPS_teacher), where 1 means the student recovers the teacher's entire edge
over its own hard-label control and negative values mean distillation hurt.
Latency is µs/row on the benchmark hold-out (median of 5 passes for
students; single timed pass for teachers, whose per-pass cost is minutes).

**Pre-registration and conduct.** Hypotheses, gates, and decision criteria
were registered before execution; deviations are dated addenda; negative
results are reported with the same prominence as positives; and one labeling
erratum discovered mid-study (a config key pointing to the wrong OpenML id)
is documented with its full rectification trail — the accidental duplicate
run it produced doubles as an exact end-to-end reproducibility check of the
pipeline. <!-- docs/desenho-experimental-destilacao.md; docs/errata-diamonds-kin8nm.md -->

## 4. Experimental Setup

Five regression datasets from the companion benchmark (wine_quality,
california_housing, superconduct, kin8nm, year_prediction; pools 5.2k–80k)
plus the complete eligible pool of fifteen from the OpenML-CTR23 suite
under verified constraints
(3k–100k rows; two suite datasets excluded for target leakage discovered
during verification — the target is a sum of features in brazilian_houses
and wave_energy; version-pinned dataset ids). Splits are identical to the
companion benchmark: a fixed 20% hold-out (seed 42) that no selection step
ever observes; three student seeds throughout. Hardware: a single RTX 5080
(16 GB); VRAM ceilings are treated as measurements, not nuisances (§5.3).
<!-- configs/datasets.yaml (regression_extension); scripts/extension_ctr23.py -->

## 5. Results

### 5.1 Point distillation fails at scale — with any teacher

Across all 20 datasets, the point student trained on teacher means beats
its hard-label control only exceptionally (2 of the first 15 evaluated); with the far stronger TabFM teacher
(anchor gaps up to 4× larger, e.g. 0.031 vs. 0.008 RMSE on
california_housing) the picture is unchanged — the best cell reaches
retention 0.21 and the rest are at or below zero. A pool-size sweep (caps
800/2k/8k) shows a genuine small-data gain on wine_quality (Δ = +0.026 at
n=800) that vanishes by n=2,000 and does not replicate on
california_housing; we report it as an observation with a dataset-dependent
moderator, not a general effect. Conclusion: at realistic pool sizes, a
tuned GBDT extracts as much point accuracy from the raw labels as from the
teacher's opinions of them.
<!-- results/distillation/distill.csv, distill_cap*.csv, extension.csv -->

### 5.2 Distributional distillation works — where the teacher has an edge

Transferring the teacher's quantile curves changes the outcome. Over the
complete 20-dataset pool, CRPS-gap retention is positive in 16 (median
+0.19), topping at +1.06 (pumadyn32nh — a born-again student that surpasses
its teacher), +0.64 (california_housing), +0.60 (abalone), +0.52
(fps_benchmark) and +0.50 (cpu_activity). Both frequentist instruments
reject at the 5% level (one-sided sign test p = 0.006; Wilcoxon p = 0.045)
and the Bayesian signed-rank places 0.70 posterior mass on positive
retention at ROPE ±0.05 (0.63–0.71 across ROPE 0.02–0.10). The four
failures are diagnosable and form the basis of §5.5's decision rule: one
dataset where the teacher trails the baseline outright (wine_quality, the
failure mode our pre-registered gate anticipated) and three price-scale,
heavy-tailed targets (fifa, diamonds, health_insurance) where §5.5 shows the
teacher's apparent edge is largely reproducible by a log transform.
Calibration accompanies the transfer: on year_prediction the distilled
student's PICP80 is 0.79 against the hard control's 0.71.
<!-- results/distillation/distill.csv + extension.csv -->

### 5.3 The serving frontier: distill vs. compress vs. shrink

We measured accuracy and latency of every strategy on three
positive-gap datasets (Fig. 1): TabPFN with context subsampled to
{1k, 5k, 10k, 25k, full}; TabPFN with 1/4/8 ensemble members; the distilled
and control students; and the TabFM point anchor.
<!-- results/distillation/pareto.csv; results/figures/pareto_distill.pdf -->

Three regimes emerge. For **point** accuracy, the hard-label student already
occupies the fast regime (2.6–5.3 µs/row) — §5.1 made visible. For
**distributional** serving, below ~250 µs/row (california_housing) and
~1,300 µs/row (year_prediction) the students are alone, and the distilled
one is CRPS-optimal among them: 0.109 at 99 µs/row where the cheapest
teacher configuration needs 245 µs/row for 0.094 (california), and 4.43 at
169 µs/row where the teacher needs 10,989 µs/row to do better (year). The
TabFM anchor holds the accuracy extreme at 18,159–394,301 µs/row.

Two side-findings have independent value. Reducing the TabPFN ensemble from
8 to 1 member *improved* RMSE on california_housing (0.2521 vs. 0.2556) at
1/7 the latency — ensemble reduction is an underrated first lever. And the
last context doubling (25k→50k) on year_prediction requires the library's
memory-saving mode on a 16 GB GPU and costs 3.3× the latency (148.6 vs.
45.1 ms/row) for ΔRMSE of 0.05 — on commodity GPUs, the full-context teacher
does not pay for itself.

### 5.4 Why out-of-fold labeling matters: calibration, not accuracy

Classification results in prior work suggest in-sample teacher labels
collapse toward one-hot and must be avoided. Regression behaves differently.
Our three-regime ablation — OOF (80% context, leak-free), in-sample at the
*same* 80% context (leaky), and in-sample at full context — shows that
leakage alone leaves RMSE and CRPS intact (it even helps slightly), that
context size contributes almost nothing, and that the harm is concentrated
in **coverage**: at identical context, in-sample-labeled students lose 0.5
to 5.1 points of PICP80 across all four datasets tested. The teacher is
overconfident on rows it has in context, and that overconfidence — invisible
to accuracy metrics — transfers to the student as too-narrow intervals. OOF
labeling in regression is therefore a calibration safeguard.
<!-- results/distillation/ablation_insample.csv (3 regimes, 72 rows) -->

### 5.5 When to distill: two refuted hypotheses become a decision rule

*Does the student family matter?* Greatly. The XGBoost pinball objective is
a weak distributional learner, and distillation reliably repairs it. A
hard-label MLP-quantile student, by contrast, matches the distilled XGBoost
with no teacher at all on small numeric datasets (CRPS 0.110 vs. 0.109 on
california_housing; 0.047 vs. 0.064 on kin8nm) — yet collapses on
year_prediction (CRPS 9.66, near-total overcoverage), where distillation
rescues it (→4.91). <!-- results/distillation/mlp_students.csv -->

*Is heavy tail the failure moderator?* We re-ran teacher and students under
y′ = log1p(y) on the three failing price-scale datasets plus two positive
controls. The transform does **not** rescue the failures (fifa persists at
−1.67) — instead it collapses the teacher's own edge (CRPS gaps shrink to
0.004–0.025 in log space) and shrinks the successes too. Much of the TFM's
distributional advantage on raw price-scale targets *is* scale handling,
reproducible for free. <!-- results/distillation/ablation_logtarget.csv -->

Together these yield the paper's practical rule:

> **Before distilling: (i) apply cheap target transforms; (ii) try a strong
> native quantile student. Distill when a genuine teacher edge persists
> after both — one OOF labeling pass, costing minutes, measures that edge
> directly.** Where the edge persists, distillation is the only route to
> sub-millisecond distributional serving (§5.3).

## 6. Discussion

**What distillation transfers.** Not the mean — the *shape*. The consistent
pattern across 15 datasets, two teachers, two student families and three
ablations is that point information saturates from raw labels, while
distributional information (quantile structure, calibrated width) is
teacher-borne and transferable — exactly the component a hard-label student
cannot recover and the component priced most steeply by in-context serving.

**Point leader ≠ distributional leader.** TabFM leads every accuracy table
in the companion benchmark yet contributes nothing to the distributional
axis: its regression head is architecturally point-only, and its ensemble
spread is measurably useless as an uncertainty estimate (16% coverage at
80% nominal). TabPFN's bar-distribution head emerges as a genuine
architectural moat, and our results argue that future TFMs should ship
distributional heads — we quantify what their absence costs.

**Serving guidance.** On 16 GB-class hardware: reduce the ensemble first
(often free), compress context second, and reserve the full-context teacher
for accuracy-critical batch settings; when distributional predictions must
be served fast, the distilled quantile student is currently the only
occupant of that regime.

## 7. Limitations

Twenty datasets support significance at the 5% level on both instruments,
but magnitudes remain heterogeneous (retention −1.3 to +1.1) and the
Bayesian posterior leaves ~25% mass on negative retention — the decision
rule of §5.5, not a universal claim, is the honest deliverable. Students inherit benchmark-tuned capacity without per-target
re-tuning. CRPS is approximated on a 19-quantile grid. TabFM results refer
to the pinned v1.0.1 release, weeks old at the time of writing and without
a peer-reviewed reference. Single-GPU measurements bound, rather than
survey, the hardware space.

## 8. Reproducibility

All code, configs, seeds, per-experiment artifacts, the pre-registered
design with dated addenda, and two public corrections (a measurement-claim
retraction and a dataset-labeling erratum whose accidental duplicate run
reproduced the pipeline exactly) are in the public repository. Weights of
both teachers are pinned; the 16 GB VRAM ceilings and their mitigations are
documented as part of the method.

## 9. Conclusion

Tabular foundation models ended the accuracy tie; their serving cost is now
the binding constraint, and distillation attacks it selectively. In
regression, what survives the transfer is the distribution — up to 64% of
the teacher's CRPS edge at microsecond latency — under conditions a
practitioner can verify in minutes. The rest of the teacher's advantage
either saturates from raw labels, dissolves under a log transform, or waits
on architectures that expose what their training already knows.
