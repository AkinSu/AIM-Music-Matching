# Lovify — local development.
#
# The short list of things people actually run. `make` on its own prints it.

.DEFAULT_GOAL := help
.PHONY: help up down logs shell db test test-host seed fresh build ps

COMPOSE := docker compose

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  First time here? docs/LOCAL_SETUP.md"

up:  ## start the stack (app on :8080, db on :5433)
	@test -f .env || { echo "No .env — copy .env.example to .env first."; exit 1; }
	$(COMPOSE) up -d
	@echo
	@echo "  app  http://localhost:8080"
	@echo "  db   postgresql://lovify:lovify@localhost:5433/lovify"
	@echo "  logs make logs"

down:  ## stop everything (data survives)
	$(COMPOSE) down

logs:  ## tail the app logs
	$(COMPOSE) logs -f web

ps:  ## what is running
	$(COMPOSE) ps

build:  ## rebuild the image
	$(COMPOSE) build

shell:  ## shell inside the web container
	$(COMPOSE) exec web bash

db:  ## psql into the database
	$(COMPOSE) exec db psql -U lovify -d lovify

seed:  ## load the audio feature store into Postgres (slow, resumable)
	$(COMPOSE) exec web python /app/scripts/seed_db.py

test:  ## run the ml test suite inside the container
	$(COMPOSE) exec -T -w /app/ml/tests web python -W ignore -m unittest discover -s . -t .

test-host:  ## run the ml test suite on the host (needs ml/.venv)
	cd ml/tests && ../.venv/bin/python -W ignore -m unittest discover -s . -t .

fresh:  ## nuke the database and start over, then reseed
	$(COMPOSE) down -v
	$(COMPOSE) build
	$(COMPOSE) up -d
	@echo "waiting for the database…"
	@until $(COMPOSE) exec -T db pg_isready -U lovify -d lovify >/dev/null 2>&1; do sleep 1; done
	$(MAKE) seed
