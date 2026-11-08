# LEGAL-UQA

LEGAL-UQA is a parallel English–Urdu constitutional question-answering dataset with 619 question–answer pairs from 305 article chunks. This repository contains the dataset, source documents, saved model outputs, data creation and inference notebooks, and offline evaluation code.

## Setup

```sh
make setup
make env                 # then add provider keys to .env
make lab                 # open JupyterLab
```

`make extras` installs optional dependencies for OCR, local model serving, and other resource-intensive workflows. The core requirements are in `requirements.txt`.

## Run inference

```sh
make qa-openai
make qa-claude
make qa-deepseek
make qa-mistral
make qa-gemma
```

Translation notebooks have corresponding `tr-*` targets. These commands execute notebooks in place and may call paid providers. Gemma QA expects a local vLLM server at `http://localhost:8002/v1` unless `VLLM_BASE_URL` and `VLLM_MODEL` override the defaults. Inspect a notebook before running it. `make run NB=path/to/notebook.ipynb` executes any notebook.

## Evaluate saved outputs

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-evaluation-lock.txt
.venv/bin/python -m nltk.downloader punkt_tab wordnet
make evaluate
make test
```

`make evaluate` recomputes dataset statistics, QA metrics, human score summaries, BM25 retrieval, and judge agreement from saved artifacts. It makes no model or API calls. Results and source hashes are written as JSON under `output/evaluation/`; the output directory is git-ignored. The lock file describes the tested Python 3.13 environment. Required NLTK resource hashes are recorded in `data/evaluation_environment.json`.

The saved mT5 response CSV and annotated workbook contain different predictions for 110 of 124 items; evaluation keeps those variants separate. Human score corrections are logged in `data/human_evaluations_qa`.

## Repository layout

- `data/complete_dataset`: train and validation data in Arrow format.
- `data/model_qa_responses`: saved model answers.
- `data/human_evaluations_qa`: human grades, corrections, and summaries.
- `data/automated_evaluations_qa/jev`: saved judge verdicts and comparisons.
- `notebooks`: document processing, dataset creation, translation, inference, and retrieval workflows.
- `scripts/evaluate_saved.py`: offline dataset and answer evaluation.
- `scripts/lexical_retrieval.py`: BM25 passage retrieval.
- `scripts/jev`: judge inference and agreement tools.

See [the data statement](data/README.md) and [judge documentation](scripts/jev/README.md). Project code and author-contributed data use MIT. Third-party documents, models, and assets retain their own rights and licenses.
