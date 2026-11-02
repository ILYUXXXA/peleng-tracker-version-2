.PHONY: install run test lint format frontend clean

install:
	uv sync --extra dev
	cd web && npm ci

run:
	uv run peleng-tracker run --config config/hackathon.yaml

test:
	uv run pytest

lint:
	uv run ruff check peleng_tracker tests
	cd web && npm run lint

format:
	uv run ruff format peleng_tracker tests

frontend:
	cd web && npm run build

clean:
	rm -rf build dist web/dist peleng_tracker/static .pytest_cache .ruff_cache
	find peleng_tracker tests -type d -name __pycache__ -prune -exec rm -rf {} +
