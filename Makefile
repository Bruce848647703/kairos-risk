PY ?= python3
DATA_DIR ?= ../kairos-data/data/ashare

.PHONY: help bootstrap test lint format clean demo real
help:
	@echo "targets: bootstrap test lint format clean demo real"
	@echo "  real: 用真实 A 股行情生成 research/real_risk 报告（DATA_DIR=$(DATA_DIR)）"

bootstrap:
	$(PY) -m venv .venv
	. .venv/bin/activate && $(PY) -m pip install -U pip && pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "pip install ruff 后再 lint"; fi

format:
	@if command -v ruff >/dev/null 2>&1; then ruff format .; else echo "pip install ruff 后再 format"; fi

demo:
	$(PY) examples/demo.py

real:
	$(PY) examples/real_risk_report.py --data-dir $(DATA_DIR)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist
