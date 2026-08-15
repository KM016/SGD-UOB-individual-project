PYTHON ?= python3

.PHONY: install install-notebook test lint smoke reproduce figures

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-notebook:
	$(PYTHON) -m pip install -e ".[dev,notebook]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check scripts tests

smoke:
	$(PYTHON) scripts/experiment.py --quick --output-root .artifacts/smoke/well-conditioned
	$(PYTHON) scripts/ill_conditioned_experiment.py --quick --output-root .artifacts/smoke/ill-conditioned

figures:
	$(PYTHON) scripts/prox_step_visualization.py
	$(PYTHON) scripts/variance_reduction_trajectory.py

reproduce:
	$(PYTHON) scripts/experiment.py
	$(PYTHON) scripts/ill_conditioned_experiment.py
	$(MAKE) figures
