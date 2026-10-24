PYTHON ?= .venv/bin/python
TYPST ?= typst
EXPERIMENTS := $(patsubst %/,%,$(dir $(wildcard experiments/[0-9][0-9]_*/run.py)))

.PHONY: help docs figures theory-figures compute summarize check smoke all summaries list clean

help:
	@echo 'make docs                      the document, docs/notes.pdf (Spanish, Typst)'
	@echo 'make figures                   figures of every experiment and the synthetic figures of the document'
	@echo 'make theory-figures            only the synthetic figures of the document (docs/figures/figures.py)'
	@echo 'make compute E=NN_name [ARGS=] one stage of one experiment (also summarize, check)'
	@echo 'make summaries                 summarize every experiment (those without raw results keep their summary)'
	@echo 'make smoke                     every stage of every experiment with --smoke (tiny-llama, CPU, results under .temp/smoke/)'
	@echo 'make all                       every experiment in order: compute, summarize (needs GPUs)'
	@echo 'make list                      list the experiments'
	@echo 'make clean                     remove caches'

docs: docs/notes.pdf

.SECONDEXPANSION:
docs/notes.pdf: docs/notes.typ docs/references.bib docs/lib.typ $$(wildcard docs/figures/*.pdf) $$(wildcard experiments/*/figures/*.pdf)
	$(TYPST) compile --root . $< $@

figures: theory-figures
	@for e in $(EXPERIMENTS); do grep -q '^def figures' $$e/run.py && $(PYTHON) src/run.py $$e figures || true; done

theory-figures:
	$(PYTHON) docs/figures/figures.py

compute summarize check:
	@test -n "$(E)" || (echo 'usage: make $@ E=NN_name [ARGS="--key value ..."]' && exit 1)
	$(PYTHON) src/run.py $(E) $@ $(ARGS)

smoke:
	@for e in $(EXPERIMENTS); do \
		$(PYTHON) src/run.py $$e compute --smoke --force && $(PYTHON) src/run.py $$e summarize --smoke || exit 1; \
		for s in figures check; do if grep -q "^def $$s" $$e/run.py; then $(PYTHON) src/run.py $$e $$s --smoke || exit 1; fi; done; \
	done

summaries:
	@for e in $(EXPERIMENTS); do $(PYTHON) src/run.py $$e summarize || exit 1; done

all:
	@for e in $(EXPERIMENTS); do $(PYTHON) src/run.py $$e compute && $(PYTHON) src/run.py $$e summarize || exit 1; done

list:
	@for e in $(EXPERIMENTS); do echo $$e; done

clean:
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -name .DS_Store -not -path './.venv/*' -delete
