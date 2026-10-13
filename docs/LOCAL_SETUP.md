# Running Lovify locally

From a fresh clone to a working app. No prior knowledge assumed.

If something here is wrong or unclear, fix it — this file is the first thing a
new person reads.

---

## Just want to look at it? Start here.

The fastest path to a working app, with **no Docker, no Spotify account, and
no credentials**:

```bash
make demo
```

Then open **http://127.0.0.1:8080/demo**.

That page gives you four fake people with synthetic libraries. The score you
get is computed by the real code — only the Spotify login and the trained
model are stubbed. Walk the whole product in about two minutes:

1. **Sign in as Ada.** You land on a waiting page with an invite link.
2. **Copy that link, open it in a private window.** That is a second device.
   Sign in as **Bo**. Both windows now show a score.
3. **Press "Create a public link".** Open the `/s/…` link it gives you in a
   third window — a stranger's view.
4. **In that third window, paste the original invite link.** You should get
   **403**. Results are for the two participants; the public link is a
   separate thing you chose to create.
5. **Press "Replace with a new link".** The first `/s/…` link now 404s.
6. **Try Dot** (20 tracks) against Ada. You get "not enough listening
   history" — a deliberate refusal, not an error.
7. **`/me/matches`** — your history, and the share controls for each one.

If `make demo` cannot find Python, it wants `ml/.venv`:

```bash
python3 -m venv ml/.venv && ml/.venv/bin/pip install -r requirements-app.txt
```

The demo refuses to start against a non-local `BASE_URL`, because it
replaces authentication with a stub. There is no `DEMO_MODE` flag on the
real app for the same reason.

### Running the tests

```bash
make test-local     # both suites on your machine, no Docker
```

Expect **125 app tests** and **237 ML tests**, all passing.

---

## Running the REAL thing, with real Spotify accounts

The demo above proves the flow. This is the actual product: your Spotify
account, your real listening history, the trained model, 1.4 million audio
records.

### 1. Check what is missing

```bash
make check
```

It tells you exactly what is not ready and how to fix each one. Run it again
after every change until it says **Ready**.

### 2. Put your Spotify credentials in `.env`

[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) →
your app → **Settings**. Copy the Client ID, press "View client secret", copy
that too.

```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

`make check` verifies them against Spotify directly, so you find out they are
wrong now rather than halfway through a login.

### 3. Register the redirect URI — this is the one that wastes an hour

Same Settings page → **Redirect URIs** → Add:

```
http://127.0.0.1:8080/callback
```

Then press **Save**, which is easy to miss.

Three ways this goes wrong, all giving the same unhelpful
`INVALID_CLIENT: Invalid redirect URI`:

- **`localhost` instead of `127.0.0.1`.** Spotify rejects `localhost`
  outright. They are the same to your browser and not to Spotify.
- **`https` instead of `http`.** Spotify requires https for every redirect
  URI *except* a loopback IP, which `127.0.0.1` is. So http is correct here
  and only here.
- **A trailing slash, or the wrong port.** It is compared character for
  character.

Whatever you register must match `BASE_URL` in `.env` exactly, plus
`/callback`. `make check` prints the exact string to paste.

### 4. Add both testers to the allowlist

Settings → **User Management** → add each person's **Spotify account email**.
**Max FIVE** while the app is in development mode — not 25, which is what this
project assumed until 2026-10-13. Both people in a match count against it, so
five users is two simultaneous pairs and one spare.
([quota modes](https://developer.spotify.com/documentation/web-api/concepts/quota-modes))

The cap is effectively permanent: since May 2025 Spotify grants extended quota
only to registered businesses with a launched service and **250,000 monthly
active users**, and does not accept individual or student applications at all.

This matters more than it sounds: an account that is not on the list gets a
**403 at login that looks exactly like a bug in the app**. If someone says
"your site is broken", check this first.

Your own account is *not* automatically on the list.

### 5. Start it

```bash
make build && make up
```

The `build` is needed once because the stack was renamed. Then open
**http://127.0.0.1:8080** — not `localhost:8080`, or the cookie and the
redirect URI will not line up.

### 6. Test it across two devices

The whole point is two people on two devices, so testing it in one browser
proves nothing:

1. Sign in on your laptop. Copy the invite link.
2. Open that link **on your phone**, or in a private window signed into the
   second Spotify account.
3. Sign in as the second person.
4. Both of you see the result.

Both accounts must be on the allowlist, and they must be genuinely different
Spotify accounts — the app refuses to match an account with itself.

### Your phone cannot be the second device (against a local server)

Your phone cannot reach `127.0.0.1:8080` on your laptop — that address means
"this machine" on whatever device asks. The obvious workaround, pointing it at
the laptop's network address like `http://192.168.1.50:8080`, does not work
either: Spotify permits plain `http` **only** for loopback literals (`127.0.0.1`
and `[::1]`), and a LAN address is not one. `localhost` is rejected outright as
well. ([redirect URI rules](https://developer.spotify.com/documentation/web-api/concepts/redirect_uri))

So, in order of effort:

1. **Two browser profiles on the laptop.** A normal window and a private one,
   each signed into a different Spotify account. This genuinely tests the
   cross-device path — the whole bug being fixed was shared cookie state, and
   two profiles have separate cookie jars.
2. **An https tunnel** if you want a real second device. `cloudflared tunnel
   --url http://127.0.0.1:8080` or `ngrok http 8080` gives you an https URL;
   register `https://<that-host>/callback` as a redirect URI and set `BASE_URL`
   to it. The URL changes each run, so you re-register each time.
3. **Deploy it.** Then the redirect URI is stable and anyone allowlisted can use
   it from anywhere.

### If something goes wrong

```bash
make logs        # what the app actually saw
make check-db    # is the feature store loaded
```

| What you see | What it means |
|---|---|
| `INVALID_CLIENT: Invalid redirect URI` | step 3. Compare character by character |
| 403 at the Spotify login screen | step 4. That account is not on the allowlist |
| "Compatibility scoring is not available" | no model artifact — `ml/artifacts/lovify_models.joblib` |
| "not enough listening history" | real, and not an error. Under 100 resolvable tracks the score is biased by tens of points, so the app refuses rather than guessing |
| "we could not read that music" | that account's music is too new or too obscure for the pre-2021 audio database |

---

## Docker reference

Needed for real Spotify logins, Postgres, and the 1.4M-row feature store.

### What you need

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

**Spotify allows only 5 users.** In development mode, every person who logs in
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
