# No production data is bundled. Run from the repository root.
VENV_PY := $(firstword $(wildcard .venv/Scripts/python.exe) $(wildcard .venv/bin/python))
PY := $(if $(VENV_PY),$(VENV_PY),python)
MASS ?= 200
STUDY ?= cg_bbc
OUT ?= $(or $(HEPML_OUTPUT_ROOT),outputs)
export HEPML_OUTPUT_ROOT := $(OUT)
CLI := $(PY) -m hepml.cli
STUDY_FLAG := --study studies/$(STUDY)
SPLITS_DIR := $(OUT)/$(STUDY)/datasets/splits

.PHONY: help prepare train ablation optimal freeze infer summarize plots ci test smoke lint package paths
help:
	@echo "make prepare/train/plots/freeze/infer/summarize after downloading compact data"
	@echo "make ci/test/smoke use synthetic temporary data; make package creates the server ZIP"
prepare:
	$(CLI) prepare $(STUDY_FLAG) --write-splits --mass $(MASS)
train:
	$(CLI) train $(STUDY_FLAG) --mass $(MASS)
ablation:
	$(CLI) ablation $(STUDY_FLAG) --mass $(MASS) --mode drop1
optimal:
	$(CLI) optimal $(STUDY_FLAG) --mass $(MASS)
plots:
	$(CLI) plots $(STUDY_FLAG) --mass $(MASS)
freeze:
	$(CLI) freeze $(STUDY_FLAG) --mass $(MASS)
infer:
	$(CLI) predict $(STUDY_FLAG) --mass $(MASS) --input $(SPLITS_DIR)/test_sig$(MASS).parquet --split test
summarize:
	$(CLI) summarize $(STUDY_FLAG) --mass $(MASS)
paths:
	$(CLI) paths $(STUDY_FLAG)
package:
	$(PY) scripts/package_server.py $(STUDY_FLAG)
lint:
	$(PY) -m ruff check src tests studies scripts compactor
ci:
	$(MAKE) lint
	$(MAKE) test
test:
	$(PY) -m pytest -q
smoke:
	$(PY) -m pytest -q -m slow
