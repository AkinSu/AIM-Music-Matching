# Lovify web service.
#
# Two stages so the runtime image carries no compiler and no pip cache. Layer
# order is deliberate: dependencies are copied and installed before any source,
# so editing a route rebuilds in seconds instead of reinstalling scikit-learn.
#
# Works unchanged for local Compose and for Railway. Railway injects PORT and
# DATABASE_URL; both are read from the environment with sane local defaults.

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — builder
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is here and NOT in the runtime stage. That is the point of
# splitting: wheels that need a compiler get one, the shipped image does not.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Everything lands in a venv so stage 2 can take it with a single COPY.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first, source later. This layer is cached until the pins change.
COPY requirements-app.txt .
RUN pip install --upgrade pip && pip install -r requirements-app.txt

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — runtime
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # ml/src on the path so `import lovify_ml` resolves without installing it
    PYTHONPATH=/app/ml/src

# libpq5 is the runtime half of the Postgres client. psycopg[binary] bundles
# its own, but keeping this makes a later switch to psycopg[c] a one-line change.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root. Created before COPY so ownership is set in one layer.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# The app directory has spaces in its name, hence the JSON form of COPY.
# It is renamed to /app/web here so nothing downstream has to quote paths.
COPY --chown=app:app ["LOVIFY ME APP/", "/app/web/"]
COPY --chown=app:app ml/src /app/ml/src
COPY --chown=app:app ml/scripts /app/ml/scripts
COPY --chown=app:app scripts /app/scripts

USER app
WORKDIR /app/web

EXPOSE 8080

# Railway sets PORT; locally it defaults to 8080. Shell form so ${PORT} expands.
# Two workers: enough for a demo, small enough that each loading a copy of the
# model does not exhaust an 8GB laptop.
CMD gunicorn --bind "0.0.0.0:${PORT:-8080}" --workers 2 --timeout 120 lovify:app


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — dev
# ═══════════════════════════════════════════════════════════════════════════
# Runtime plus a test runner. Compose builds this (`target: dev`); a plain
# `docker build .` stops at `runtime` and ships without pytest, which is the
# point of making it a separate stage rather than a line in requirements-app.
#
# It exists because the app suite had no way to run inside the container, so
# the one seam that matters -- the app calling the real model against the real
# database -- was only ever exercised by a stub.
FROM runtime AS dev

USER root
COPY requirements-app.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
USER app
WORKDIR /app/web
