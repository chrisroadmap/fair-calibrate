# Gratitude to IIASA Climate Assessment, which this is based upon

.DEFAULT_GOAL := help

VENV_DIR ?= ./venv

FILES_TO_FORMAT_PYTHON=src tests setup.py

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([0-9a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: test
test: $(VENV_DIR)  ## run all the tests
	$(VENV_DIR)/bin/pytest tests -r a

.PHONY: checks
checks: $(VENV_DIR)  ## run all the checks
	@echo "=== bandit ==="; $(VENV_DIR)/bin/bandit -c .bandit.yml -r src/fair21 || echo "--- bandit failed ---" >&2; \
		echo "\n\n=== black ==="; $(VENV_DIR)/bin/black --check src tests setup.py --exclude fair21/_version.py || echo "--- black failed ---" >&2; \
		echo "\n\n=== isort ==="; $(VENV_DIR)/bin/isort --check-only --quiet src tests setup.py || echo "--- isort failed ---" >&2; \
		echo "\n\n=== flake8 ==="; $(VENV_DIR)/bin/flake8 src tests setup.py || echo "--- flake8 failed ---" >&2; \
		echo

.PHONY: format
format:  ## re-format files
	make isort
	make black

.PHONY: black
black: $(VENV_DIR)  ## use black to autoformat code
	@status=$$(git status --porcelain); \
	if test ${FORCE} ||test "x$${status}" = x; then \
		$(VENV_DIR)/bin/black --target-version py37 $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting, working directory is dirty... >&2; \
	fi;

isort: $(VENV_DIR)  ## format the code
	@status=$$(git status --porcelain src tests); \
	if test ${FORCE} || test "x$${status}" = x; then \
		$(VENV_DIR)/bin/isort $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

virtual-environment: $(VENV_DIR)  ## update venv, create a new venv if it doesn't exist
$(VENV_DIR): setup.py
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade 'pip>=20.3'
	$(VENV_DIR)/bin/pip install wheel
	$(VENV_DIR)/bin/pip install -e .[dev]
	$(VENV_DIR)/bin/jupyter nbextension enable --py widgetsnbextension --sys-prefix

	touch $(VENV_DIR)

first-venv: ## create a new virtual environment for the very first repo setup
	python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install versioneer
	# don't touch here as we don't want this venv to persist anyway
