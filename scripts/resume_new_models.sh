#!/usr/bin/env bash
# Retoma o benchmark dos 3 novos modelos (tabkan, kan, tabfm).
# Resumivel: pula todo (modelo x dataset) que ja tem JSON em results/raw/.
cd "$(dirname "$0")/.."
echo "--- retomada $(date '+%F %T') ---" >> results/logs/run_new_models.log
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/run_all.py -m tabkan -m kan -m tabfm \
  >> results/logs/run_new_models.log 2>&1 &
echo "relancado em background (PID $!). Acompanhe com:"
echo "  tail -f results/logs/run_new_models.log"
