.PHONY: install data train report all test datatest predict sync clean

install:
	pip install -e ".[dev]"

data:
	python -m embryofusion.cli generate

train:
	python -m embryofusion.cli train

report:
	python -m embryofusion.cli report

all:
	python -m embryofusion.cli all

test:
	pytest -q

datatest:
	pytest -q -m data

sync:
	wandb sync reports/tracking/wandb/offline-run-*

clean:
	rm -rf data/paired reports/tracking .pytest_cache
