# Makefile — one-command reproduction for tabular benchmark experiments
#
# Usage:
#   make install        Install dependencies
#   make download       Download all 15 datasets from OpenML
#   make test           Run full test suite
#   make test-fast      Run tests excluding slow markers
#   make train          Run a single experiment (MODEL=xgboost DATASET=wine_quality)
#   make run-all        Run the full experimental pipeline
#   make evaluate       Aggregate results and run statistical tests
#   make notebooks      Execute all Jupyter notebooks
#   make clean          Remove result files and logs
#   make all            Full reproduction: install -> download -> run-all -> evaluate

.PHONY: install download test test-fast train run-all evaluate notebooks clean all

# Defaults for single-experiment target
MODEL   ?= xgboost
DATASET ?= wine_quality
GPU     ?=

# Build GPU flag only if GPU is set
ifdef GPU
  GPU_FLAG := --gpu $(GPU)
else
  GPU_FLAG :=
endif

install:
	uv sync

download:
	uv run python scripts/download_data.py

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow"

train:
	uv run python scripts/train.py --model $(MODEL) --dataset $(DATASET) $(GPU_FLAG)

run-all:
	uv run python scripts/run_all.py $(GPU_FLAG)

evaluate:
	uv run python scripts/evaluate.py

notebooks:
	@for nb in notebooks/*.ipynb; do \
		echo "Executing $$nb..."; \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

clean:
	rm -f results/raw/*.json
	rm -f results/logs/*.log
	rm -f results/aggregated/*.csv
	rm -f results/figures/*.png results/figures/*.pdf
	rm -f results/environment.json

all: install download run-all evaluate
