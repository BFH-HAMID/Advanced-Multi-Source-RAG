# Advanced Multi-Source RAG — developer shortcuts
.PHONY: help install samples index index-fast app ask eval test lint format docker docker-run clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:           ## Create .venv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "✔ dependencies installed — edit .env to pick your LLM provider"

samples:           ## Generate the demo knowledge base (PDF/CSV/Bangla/MD + test set)
	$(PY) scripts/generate_samples.py

index:             ## Ingest demo data and build all three indexes
	$(PY) scripts/build_index.py

index-fast:        ## Build indexes without the knowledge graph
	$(PY) scripts/build_index.py --skip-graph

app:               ## Run the Streamlit UI on http://localhost:8501
	$(PY) -m streamlit run app/streamlit_app.py

ask:               ## Ask a question: make ask Q="How many annual leave days?"
	$(PY) scripts/ask.py "$(Q)"

eval:              ## Run the RAGAS / heuristic evaluation
	$(PY) scripts/run_eval.py

test:              ## Run the test-suite
	$(PY) -m pytest tests -q

cov:               ## Run tests with coverage
	$(PY) -m pytest tests --cov=src --cov-report=term-missing

lint:              ## Lint with ruff
	$(PY) -m ruff check src app scripts tests

format:            ## Auto-fix lint issues
	$(PY) -m ruff check --fix src app scripts tests

docker:            ## Build the Docker image
	docker build -t enterprise-rag:latest .

docker-run:        ## Run the container with the local .env and data volume
	docker run --rm -p 8501:8501 --env-file .env -v "$(PWD)/data:/app/data" enterprise-rag:latest

clean:             ## Remove generated indexes, DB and caches (keeps raw sources)
	rm -rf data/indexes data/db data/processed/* .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
