.PHONY: dev test lint data preflight smoke train eval-zero eval clean

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

data:
	python scripts/download_data.py --out data

preflight:
	python scripts/preflight.py --config configs/server.yaml

# Minutes. Proves the pipeline before committing GPU hours.
smoke:
	python -m simulst.train --config configs/smoke.yaml

# The number any fine-tune must beat.
eval-zero:
	python -m simulst.evaluate --config configs/server.yaml

train:
	python -m simulst.train --config configs/server.yaml

eval:
	python -m simulst.evaluate --config configs/server.yaml --checkpoint runs/whisper-small-ft/best

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
