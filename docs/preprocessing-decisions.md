# Preprocessing & Coverage Decisions

This note documents three methodological decisions made after the first full
benchmark run surfaced 10 failed (model, dataset) cells. All failures were
CUDA out-of-memory or hard architectural limits — none were logic bugs. The
decisions below are the principled fixes, applied consistently.

## 1. One-hot cardinality cap for deep-learning preprocessing

**Problem.** The deep-learning family one-hot encodes categoricals. Two datasets
contain ID-like categorical columns with thousands of levels:

| Dataset | Offending column(s) | Levels | One-hot width (before) |
|---------|--------------------|--------|------------------------|
| amazon_employee | RESOURCE, MGR_ID (+7 more) | 7,518 / 4,243 | 6,910 columns |
| telco_customer_churn | customerID | 6,531 | 4,759 columns |

Attention-based models (FT-Transformer, SAINT, STab) create one token per input
column, so the attention matrix scales with (width)². With ~7K columns this
required 50–100 GiB and OOM'd on a 16 GB GPU. Plain MLP-based models (MLP, TabM,
TabNet) survived because they only widen the input layer.

**Decision.** Cap one-hot at `max_categories=50` with
`handle_unknown="infrequent_if_exist"`. Columns with >50 levels keep the 49 most
frequent and collapse the rest into one "infrequent" bucket; columns with ≤50
levels are **unaffected** (identical output).

**Why 50.** The cardinality distribution has a clean gap: the largest "moderate"
categorical in the whole suite is adult's `native_country` at 41 levels; the next
group jumps to 6,500+. Any threshold in (41, 6500) separates moderate categoricals
from ID-like explosions. 50 is a round value above 41, so:

- **adult, bank_marketing, credit_g, covertype, cardiovascular, superconduct** —
  all categoricals ≤41 levels → encoding unchanged → results **not** re-run.
- **amazon_employee** (6,910→450 cols) and **telco** (4,759→94 cols) — capped →
  all 7 DL models re-run on these two datasets for consistency.

GBDT and TabPFN preprocessing (ordinal encoding) is untouched, so their results
on amazon_employee/telco remain valid and were not re-run.

One-hot of high-cardinality categoricals is a known anti-pattern; no serious
tabular-DL paper uses it. The cap is the standard, defensible fix and the
behaviour it replaces (naive one-hot OOM) is itself a reportable finding.

## 2. TabPFN batched inference

**Problem.** TabPFN holds the training set as in-context memory and scores the
whole test set in one forward pass. On large regression test sets
(year_prediction: 16K test rows; superconduct: 6.3K) against a 50K-sample
context this OOM'd at inference.

**Decision.** Chunk `predict`/`predict_proba` into row-batches of 2,000 and
concatenate. Predictions are independent across rows, so results are identical;
only peak memory drops. tabpfn × {year_prediction, superconduct} re-run.

## 3. TabPFN on helena — N/A (architectural limit)

helena is a 100-class problem. TabPFN officially supports **at most 10 classes**;
it raises an error above that. This is a fundamental model constraint, not a bug
and not fixable. **tabpfn × helena is reported as N/A** — not every model is
applicable to every task, and this is itself a documented limitation of the
foundation-model family.

## Re-run scope

16 experiments: 7 DL × {amazon_employee, telco_customer_churn} + tabpfn ×
{year_prediction, superconduct}. Run with
`PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. After
this, coverage is 197/198 with one principled N/A (tabpfn × helena).
