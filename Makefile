PYTHON ?= python3
CONFIG ?= python/configs/fashiongen_experiments.yaml

.PHONY: api web compare evaluate tsne ann demo writeup-assets writeup-pdf experiment-suite

api:
	cd python && $(PYTHON) -m uvicorn app.main:app --reload

web:
	cd nextjs && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev

compare:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks compare

evaluate:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks evaluate

tsne:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks tsne

ann:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks ann

demo:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks demo

writeup-assets:
	cd python && $(PYTHON) scripts/prepare_report_assets.py

writeup-pdf:
	cd python && $(PYTHON) scripts/prepare_report_assets.py
	cd reports && latexmk -pdf -interaction=nonstopmode -halt-on-error progress_report.tex

experiment-suite:
	cd python && $(PYTHON) scripts/run_report_suite.py --config configs/fashiongen_experiments.yaml --tasks compare evaluate tsne ann
