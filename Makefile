# Lovify — local development.
#
# The short list of things people actually run. `make` on its own prints it.

.DEFAULT_GOAL := help
.PHONY: help up down logs shell db test test-app test-all test-host test-local \
        seed sweep demo check check-db fresh build ps

COMPOSE := docker compose

# What `up` prints has to be what is actually bound. These used to be
# literals, so anyone who followed docs/LOCAL_SETUP.md's advice to change
# WEB_HOST_PORT after a port clash was then told to open the wrong URL.
#
# Read with sed rather than `include .env`: that file holds a secret key and
# whatever else people put in it, and make would parse all of it, expand any
# `$` it found, and fail on anything that is not KEY=VALUE. One grep for one
# key is the whole need.
envdef = $(shell sed -n 's/^$(1)=//p' .env 2>/dev/null | tail -1)
WEB_HOST_PORT := $(or $(call envdef,WEB_HOST_PORT),8080)
DB_HOST_PORT  := $(or $(call envdef,DB_HOST_PORT),5433)
PG_USER       := $(or $(call envdef,POSTGRES_USER),lovify)
PG_DB         := $(or $(call envdef,POSTGRES_DB),lovify)

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  First time here? docs/LOCAL_SETUP.md"

up:  ## start the stack (ports come from .env)
	@test -f .env || { echo "No .env — copy .env.example to .env first."; exit 1; }
	$(COMPOSE) up -d
	@echo
	@echo "  app  http://localhost:$(WEB_HOST_PORT)"
	@echo "  db   postgresql://$(PG_USER)@localhost:$(DB_HOST_PORT)/$(PG_DB)"
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
	$(COMPOSE) exec db psql -U $(PG_USER) -d $(PG_DB)

seed:  ## load the audio feature store into Postgres (slow, resumable)
	$(COMPOSE) exec web python /app/scripts/seed_db.py

test:  ## run the ml test suite inside the container
	$(COMPOSE) exec -T -w /app/ml/tests web python -W ignore -m unittest discover -s . -t .

test-app:  ## run the app test suite inside the container (real Postgres)
	@# A SEPARATE database. The suite drops every table before each test, and
	@# the development database holds the 1.4M-row seeded feature store.
	@$(COMPOSE) exec -T db psql -U $(PG_USER) -d $(PG_DB) -tc \
	  "SELECT 1 FROM pg_database WHERE datname='$(PG_DB)_test'" | grep -q 1 \
	  || $(COMPOSE) exec -T db createdb -U $(PG_USER) $(PG_DB)_test
	@# The scratch database is where match state is written and wiped. The
	@# seeded one is passed separately and only ever read, which is what lets
	@# tests/test_real_model.py exercise the trained model against real audio
	@# features instead of a stub. Without it those seven tests skip.
	$(COMPOSE) exec -T -w /app/web \
	  -e DATABASE_URL=postgresql+psycopg://$(PG_USER):lovify@db:5432/$(PG_DB)_test \
	  -e LOVIFY_SEEDED_DATABASE_URL=postgresql+psycopg://$(PG_USER):lovify@db:5432/$(PG_DB) \
	  web python -m pytest -q tests

test-all:  ## both suites, the way CI runs them
	$(MAKE) test
	$(MAKE) test-app

# Pick an interpreter up front. A `cmd || fallback` here would re-run the
# whole check whenever it reported a problem, because "problems found" and
# "interpreter missing" both exit non-zero.
PY := $(shell test -x ml/.venv/bin/python && echo ml/.venv/bin/python || echo python3)

check:  ## is the REAL app ready to run? (credentials, model, data)
	@$(PY) scripts/preflight.py

check-db:  ## feature store row count, from inside the stack
	@$(COMPOSE) exec -T db psql -U $(PG_USER) -d $(PG_DB) -tAc \
	  "SELECT count(*) || ' tracks in the feature store' FROM track_features"

demo:  ## click through the whole match flow, no Spotify account needed
	@# Deliberately not a flag on the real app: a DEMO_MODE env var that
	@# stubs out authentication is one mistake away from being set in
	@# production. This is a separate entry point that refuses to run
	@# against a non-local BASE_URL.
	ml/.venv/bin/python scripts/demo.py

sweep:  ## delete expired matches and old track lists (run daily in deployment)
	$(COMPOSE) exec -T -w /app/web web python retention.py

test-local:  ## both suites on the host, no Docker (needs ml/.venv)
	cd ml && .venv/bin/python -m pytest -q tests
	cd "LOVIFY ME APP" && SECRET_KEY=test-only ../ml/.venv/bin/python -m pytest -q tests

test-host:  ## run the ml test suite on the host (needs ml/.venv)
	cd ml/tests && ../.venv/bin/python -W ignore -m unittest discover -s . -t .

fresh:  ## nuke the database and start over, then reseed
	$(COMPOSE) down -v
	$(COMPOSE) build
	$(COMPOSE) up -d
	@echo "waiting for the database…"
	@until $(COMPOSE) exec -T db pg_isready -U lovify -d lovify >/dev/null 2>&1; do sleep 1; done
	$(MAKE) seed
