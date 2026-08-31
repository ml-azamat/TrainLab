SHELL := /bin/bash
PY    := .venv/bin/python
PIP   := .venv/bin/pip
DATA  := data/imagenette2-160

# Where each process listens. Everything binds loopback by default — this is a local tool,
# not a shared service — and every value is overridable per invocation:
#
#   make api PORT=9000
#   make up-local MLFLOW_PORT=5555      (then set tracking.tracking_uri to match)
#   make ui UI_PORT=5174 API_URL=http://127.0.0.1:9000
#
# Serving the API on a non-loopback interface needs the name you browse it by named too,
# or its Host header is rejected — see TRAINLAB_ALLOWED_HOSTS below. `make api` passes
# $(HOST) through for you, so `make api HOST=192.168.1.5` works as it stands.
HOST    ?= 127.0.0.1
PORT    ?= 8000
API_URL ?= http://$(HOST):$(PORT)

# Extra hostnames the API will answer to, comma-separated, on top of localhost/127.0.0.1.
TRAINLAB_ALLOWED_HOSTS ?=

# Vite dev server (frontend work only; `make api` serves the built UI on its own port).
UI_HOST ?= 127.0.0.1
UI_PORT ?= 5173

# 5050, not MLflow's conventional 5000: macOS AirPlay Receiver listens on 5000 and 5001.
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5050

# Where the app looks for that tracker. `make api` passes it through, so moving the
# tracker with the variables above moves the form's default and the Compare tab with it.
# The `tracking.tracking_uri` field still overrides it per run.
TRACKING_URI ?= http://$(MLFLOW_HOST):$(MLFLOW_PORT)

.DEFAULT_GOAL := help
.PHONY: help setup up up-local down api ui dev build smoke clean-runs test test-py test-ui

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- setup

setup: ## Create the venv, install Python deps and build the UI
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	cd frontend && npm install --no-audit --no-fund && npm run build
	@echo ""
	@echo "Setup complete. Next:  make up-local  (tracker)  then  make api"

# ----------------------------------------------------------------- tracker

up: ## Start the MLflow stack with docker-compose (MLFLOW_HOST/MLFLOW_PORT)
	MLFLOW_HOST=$(MLFLOW_HOST) MLFLOW_PORT=$(MLFLOW_PORT) docker compose up -d
	@echo "MLflow UI:    http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	@echo "MinIO console: http://127.0.0.1:9001  (minioadmin / minioadmin)"

down: ## Stop the docker-compose stack
	MLFLOW_HOST=$(MLFLOW_HOST) MLFLOW_PORT=$(MLFLOW_PORT) docker compose down

up-local: ## Start MLflow without Docker (SQLite + local artifacts, same URL)
	@mkdir -p mlruns-local/artifacts
	@echo "MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)  (ctrl-C to stop)"
	.venv/bin/mlflow server \
	  --backend-store-uri sqlite:///$(PWD)/mlruns-local/mlflow.db \
	  --artifacts-destination $(PWD)/mlruns-local/artifacts \
	  --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

# ----------------------------------------------------------------- app

api: ## Run the FastAPI backend, serving the built UI (HOST, PORT, TRACKING_URI)
	TRAINLAB_ALLOWED_HOSTS="$(HOST),$(TRAINLAB_ALLOWED_HOSTS)" \
	  TRAINLAB_TRACKING_URI="$(TRACKING_URI)" \
	  $(PY) -m uvicorn backend.app.main:app --host $(HOST) --port $(PORT) --reload

ui: ## Run the Vite dev server with hot reload (UI_HOST, UI_PORT, API_URL)
	cd frontend && TRAINLAB_UI_HOST=$(UI_HOST) TRAINLAB_UI_PORT=$(UI_PORT) \
	  TRAINLAB_API_URL=$(API_URL) npm run dev

build: ## Production build of the UI into frontend/dist
	cd frontend && npm run build

dev: ## Reminder of the two processes needed for development
	@echo "Run these in two terminals:"
	@echo "  make api   # backend + API on $(API_URL)"
	@echo "  make ui    # Vite dev server on http://$(UI_HOST):$(UI_PORT) (proxies /api to $(API_URL))"

# ----------------------------------------------------------------- smoke test

$(DATA):
	@mkdir -p data
	@echo "Downloading Imagenette (~95 MB)…"
	curl -L --retry 3 -o data/imagenette2-160.tgz \
	  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
	tar xzf data/imagenette2-160.tgz -C data
	rm data/imagenette2-160.tgz

smoke: $(DATA) ## Seeded ~2-minute run on Imagenette exercising the whole path
	$(PY) scripts/smoke_test.py

test: ## Fast checks that need no dataset (schema, engine, data, backend, UI logic)
	$(PY) -m pytest tests -q
	cd frontend && npm test --silent

test-py: ## Python tests only
	$(PY) -m pytest tests -q

test-ui: ## Frontend unit tests only
	cd frontend && npm test

clean-runs: ## Delete local run outputs (does NOT touch the tracker)
	rm -rf runs/*
	@echo "Cleared ./runs"
