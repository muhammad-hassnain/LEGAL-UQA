# LEGAL-UQA — common tasks. Run `make help` for the list.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
QA     := notebooks/inference_notebooks_qa
TR     := notebooks/inference_notebooks_translation

.PHONY: help setup extras env lab run \
        qa-claude qa-openai qa-deepseek qa-mistral qa-gemma \
        tr-claude tr-openai tr-deepseek tr-mistral tr-gemma \
        evaluate evaluate-translation test jev-run jev-agreement

help:
	@echo "Setup:"
	@echo "  make setup       Create .venv and install core requirements"
	@echo "  make extras      Install heavy extras (torch, OCR, vLLM, ...)"
	@echo "  make env         Create .env from .env.example (then add your keys)"
	@echo "  make lab         Launch JupyterLab"
	@echo ""
	@echo "Run a QA inference notebook (writes data/model_qa_responses/*.csv):"
	@echo "  make qa-claude | qa-openai | qa-deepseek | qa-mistral | qa-gemma"
	@echo ""
	@echo "Run a translation notebook (writes data/model_translation_responses/*.csv):"
	@echo "  make tr-claude | tr-openai | tr-deepseek | tr-mistral | tr-gemma"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate       Recompute saved QA evaluations offline (no inference)"
	@echo "  make evaluate-translation  Execute the separate translation evaluation notebook"
	@echo "  make test           Run scoring and checkpoint regression checks"
	@echo "  make jev-run        Run JEV (noul mode) over every model with a human eval"
	@echo "  make jev-run MODEL=claude-sonnet-5   # one model only"
	@echo "  make jev-agreement  Summarise JEV vs human-eval agreement to CSV"
	@echo ""
	@echo "Run any notebook:"
	@echo "  make run NB=path/to/notebook.ipynb"
	@echo ""
	@echo "Note: gemma-* targets need a local vLLM server:"
	@echo "  QA: vllm serve google/gemma-4-E4B-it --port 8002"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "Done. Next: 'make env', add your keys to .env, then 'make lab'."

extras:
	$(BIN)/pip install -r requirements-extra.txt

env:
	@test -f .env || cp .env.example .env
	@echo ".env is ready — open it and fill in your API keys."

lab:
	$(BIN)/jupyter lab

# Execute a notebook in place (runs every cell, saves outputs back).
run:
	@test -n "$(NB)" || (echo "Usage: make run NB=path/to/notebook.ipynb" && exit 1)
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace "$(NB)"

qa-claude:    ; $(MAKE) run NB=$(QA)/claude_qa.ipynb
qa-openai:    ; $(MAKE) run NB=$(QA)/openai_qa.ipynb
qa-deepseek:  ; $(MAKE) run NB=$(QA)/deepseek_qa.ipynb
qa-mistral:   ; $(MAKE) run NB=$(QA)/mistral_qa.ipynb
qa-gemma:     ; $(MAKE) run NB=$(QA)/gemma_qa.ipynb

tr-claude:    ; $(MAKE) run NB=$(TR)/claude_translate.ipynb
tr-openai:    ; $(MAKE) run NB=$(TR)/openai_translate.ipynb
tr-deepseek:  ; $(MAKE) run NB=$(TR)/deepseek_translate.ipynb
tr-mistral:   ; $(MAKE) run NB=$(TR)/mistral_translate.ipynb
tr-gemma:     ; $(MAKE) run NB=$(TR)/gemma_translate.ipynb

evaluate:
	$(BIN)/python -m scripts.jev.agreement
	$(BIN)/python -m scripts.jev.threshold_sweep --no-plot
	$(BIN)/python -m scripts.evaluate_saved

evaluate-translation:
	$(MAKE) run NB=notebooks/evaluation_notebooks/translation_evaluations.ipynb

test:
	$(BIN)/python -m unittest discover -s tests -v

jev-run:
	$(BIN)/python -m scripts.jev.run $(if $(MODEL),--model $(MODEL),)

jev-agreement:
	$(BIN)/python -m scripts.jev.agreement $(if $(MODEL),--model $(MODEL),)
