.PHONY: lint typecheck test check format

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

typecheck:
	uv run mypy .

test:
	uv run pytest --cov=omnimux

check: format typecheck test
