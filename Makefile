.PHONY: test lint format eval clean

test:
	PYTHONPATH=src:. python3 -m pytest tests/ -v

lint:
	ruff check src/ eval/ tests/ scripts/

format:
	ruff format src/ eval/ tests/ scripts/

eval:
	PYTHONPATH=src:. python3 scripts/run_e2e_eval.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
