# Running Lovify locally

From a fresh clone to a working app. No prior knowledge assumed.

If something here is wrong or unclear, fix it — this file is the first thing a
new person reads.

---

## What you need

- **Docker Desktop** — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
  Everything runs inside it, so you do not need Python, Postgres, or anything
  else installed on your machine.
- **make** — already on macOS and Linux. On Windows use WSL.

That is it. Roughly 3GB of disk for the image and database.

---

## Three commands

```bash
git clone https://github.com/AkinSu/AIM-Music-Matching.git
cd AIM-Music-Matching

cp .env.example .env          # 1. create your config
python3 -c "import secrets; print(secrets.token_hex(32))"
                              #    paste the output as SECRET_KEY in .env

make up                       # 2. build and start (first build ~3 min)
make seed                     # 3. load the audio data, if you have it
```

Then open **http://localhost:8080**.

`make` on its own lists everything available.

---

## What is running

| | where | what it is |
|---|---|---|
| **web** | localhost:8080 | the Flask app, auto-reloading on edit |
| **db** | localhost:5433 | Postgres 16 |

Port **5433**, not 5432, because a lot of machines already have Postgres on the
usual port. Connect a GUI with:

```
postgresql://lovify:lovify@localhost:5433/lovify
```

Your edits to `LOVIFY ME APP/` and `ml/src/` apply immediately — the source is
mounted into the container, not baked into it. You only need `make build` when
dependencies change.

---

## The most likely first problem: no audio data

`make seed` will probably tell you this:

```
No feature store at /app/ml/data/raw/feature_store.csv
```

**This is expected on a fresh clone and is not an error.** That file is 135MB,
so it is deliberately not in git.

The app runs fine without it. What you lose is compatibility scoring, because
there are no audio features to look up — pages render, the Spotify login works,
and track lookups simply come back empty.

To build it, you need the Kaggle dump:

1. Download **"Spotify Dataset 1921-2020, 600k+ Tracks"** from
   [Kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
   (free account required).
2. Unzip it into `ml/data/raw/kaggle_600k/`.
3. Build the store:
   ```bash
   cd ml
   python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/python scripts/prepare_catalog.py
   ```
4. Back at the repo root: `make seed`

Seeding takes about 25 seconds and loads ~1.4M rows. It is safe to interrupt
and safe to re-run — it tops up whatever is missing rather than starting over.

Easier option: ask a teammate who already has `feature_store.csv` and
`tracks.csv` to send them, and drop them in `ml/data/raw/`.

---

## Spotify login

The login flow needs credentials in `.env`:

```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

from [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard).

**Two things that will catch you out:**

**The redirect URI must match exactly.** Add `http://127.0.0.1:8080/callback`
to your Spotify app's settings. Not `localhost`, not a different port — Spotify
compares the string literally and rejects anything else.

**Spotify allows only 25 users.** In development mode, every person who logs in
must be added by their Spotify account email in the dashboard first. Anyone not
on that list gets a confusing authorisation failure. For a demo with two
people, both have to be on the list.

---

## Commands

```
make up        start everything
make down      stop (your database survives)
make logs      tail the app logs
make shell     bash inside the web container
make db        psql into the database
make seed      load the audio feature store
make test      run the ML test suite in the container
make fresh     delete the database and start completely over
```

---

## When things go wrong

**`No .env — copy .env.example to .env first.`**
Exactly what it says. `cp .env.example .env`, then set `SECRET_KEY`.

**`SECRET_KEY` error on startup**
The app refuses to boot without one. That is deliberate: a default value was
once committed to this public repo, and silently falling back to a known key
would let anyone forge a login. Generate one:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

**Port already in use**
Something else is on 8080 or 5433. Change `WEB_HOST_PORT` or `DB_HOST_PORT` in
`.env` and `make up` again.

**App will not start / strange import errors**
```bash
make down && make build && make up
```

**Database looks wrong, or you want a clean slate**
```bash
make fresh
```
This destroys the database and reseeds. It does not touch your `.env` or any
file in `ml/data/`.

**Docker Desktop stopped responding**
It happens, especially during a large build. Quit and reopen Docker Desktop,
then `make up`. Containers restart themselves.

---

## Where things live

```
LOVIFY ME APP/      the Flask app — routes, templates, models
ml/src/lovify_ml/   the ML library the app imports
ml/data/raw/        large data files (gitignored, you build these)
scripts/            operational scripts, e.g. the database seeder
docker-compose.yml  the local stack
Dockerfile          how the app image is built
```

`ml/` also holds the notebooks and the research behind the models. Start with
`ml/README.md` for that side, and `ARCHITECTURE.md` at the repo root for how
the match flow works.
