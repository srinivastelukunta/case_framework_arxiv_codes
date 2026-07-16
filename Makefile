# CASE Framework Evidence Pipeline
# Windows note: if `make` is unavailable, run the equivalent
# `python -m src.run <target>` / `python -m pytest` commands directly
# (see README).

PY ?= python

.PHONY: all study1 study2 study3 refresh tables figures test

all: study1 study2 study3 tables figures

study1:
	$(PY) -m src.run study1

study2:
	$(PY) -m src.run study2

study3:
	$(PY) -m src.run study3

refresh:
	$(PY) -m src.run refresh

tables:
	$(PY) -m src.run tables

figures:
	$(PY) -m src.run figures

test:
	$(PY) -m pytest
