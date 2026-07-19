#!/usr/bin/env bash
# Turno noturno 19/07: latencia dos 3 novos -> fase 2 destilacao (TabFM
# teacher) -> varredura de tamanho de pool (hipotese do H1). Sequencial na
# GPU; cada etapa e resumivel e loga em results/logs/.
cd "$(dirname "$0")/.."
export PYTORCH_ALLOC_CONF=expandable_segments:True
L=results/logs

echo "=== [1/4] latencia kan/tabkan/tabfm $(date '+%H:%M') ==="
uv run python scripts/measure_latency.py >> $L/latency_new_models.log 2>&1
echo "latencia exit=$?"

echo "=== [2/4] OOF teacher=tabfm (5 datasets) $(date '+%H:%M') ==="
uv run python scripts/distill.py --stage oof --teacher tabfm >> $L/distill_oof_tabfm.log 2>&1
echo "oof-tabfm exit=$?"

echo "=== [3/4] students teacher=tabfm $(date '+%H:%M') ==="
uv run python scripts/distill.py --stage students --teacher tabfm >> $L/distill_students_tabfm.log 2>&1
echo "students-tabfm exit=$?"

echo "=== [4/4] varredura de pool (tabpfn, california+wine) $(date '+%H:%M') ==="
for CAP in 800 2000 8000; do
  uv run python scripts/distill.py --stage oof --teacher tabpfn --cap $CAP \
    --datasets california_housing,wine_quality >> $L/distill_sweep.log 2>&1
  uv run python scripts/distill.py --stage students --teacher tabpfn --cap $CAP \
    --datasets california_housing,wine_quality >> $L/distill_sweep.log 2>&1
  echo "cap=$CAP exit=$?"
done

echo "NIGHT_SHIFT_DONE $(date '+%F %H:%M')"
