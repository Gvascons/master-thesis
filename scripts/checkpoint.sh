#!/usr/bin/env bash
# checkpoint.sh — commit and push the latest experiment results to GitHub.
#
# Safe to run while run_all.py is running. Git operations are read-only
# from the perspective of the running pipeline — they don't modify or
# delete any files in results/raw/.
#
# Usage:
#   ./scripts/checkpoint.sh
#   ./scripts/checkpoint.sh "custom commit message"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Check we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: Not in a git repository"
    exit 1
fi

# Count completed experiments
N_COMPLETED=$(find results/raw -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)

if [ "$N_COMPLETED" -eq 0 ]; then
    echo "No experiments to checkpoint (results/raw/ is empty)"
    exit 0
fi

# Stage only the experiment results
git add results/raw/ .gitignore 2>/dev/null || true

# Check if there's anything new to commit
if git diff --cached --quiet; then
    echo "No new results to checkpoint (everything already committed)"
    echo "Current progress: $N_COMPLETED / 198 experiments"
    exit 0
fi

# Show what's being committed
echo "=== Staging these result files ==="
git diff --cached --name-only | head -20
N_STAGED=$(git diff --cached --name-only | wc -l)
if [ "$N_STAGED" -gt 20 ]; then
    echo "... and $((N_STAGED - 20)) more"
fi

# Commit
MESSAGE="${1:-checkpoint: save $N_COMPLETED / 198 experiment results}"
git commit -m "$MESSAGE"

# Push
echo "=== Pushing to GitHub ==="
git push

echo "=== Checkpoint complete: $N_COMPLETED / 198 experiments saved remotely ==="
