# Architecture

How the Lovify web app is built, and why. This is the decision record the rest of the
team builds against — if you are about to write code that contradicts something here,
change this document first and say why.

Written 2026-10-12 against the old `lovify.py` on `main` (227 lines, 9 routes), and
revised the same day against the rewrite on `agent/app-match-flow`, which is what
the tables below now describe. Where the two differ, the revision says so.

---

## 0. Blockers — both cleared

This section listed two things that had to land before the design below could be
built. Both have. It is kept rather than deleted so the branch history stays readable.

**`ml/` is on `main`** (PR #2, merged). `import lovify_ml` resolves. Note that the
`ml/artifacts`, `ml/data` and `ml/html` directories in a working tree are gitignored
residue, not tracked source — do not reason about what is on a branch from `ls`.

**`SECRET_KEY` is no longer hardcoded** (PR #1, merged). `main` now reads it from the
environment and raises on startup if it is missing, rather than falling back to a
literal published in this repository's git history. That matters here because the
match flow below puts more trust in the session cookie than the old app did.

---

## 1. The product flow

1. **A** signs in with Spotify.
2. **A** gets a share link and sends it to **B**, on a different device.
3. **B** opens it: *"Akin wants to compare music taste with you — we'll need your top
   songs from Spotify."*
4. **B** signs in with Spotify.
5. **Both** A and B see the result.

### Why the current code cannot do this

`lovify.py` keeps both participants in one browser's cookie:

```python
session['first_user_token']  = access_token      # line 174
session['first_user_info']   = {...}             # line 176
session['second_user_token'] = access_token      # line 214
session['second_user_info']  = {...}             # line 216
```

and `/find-compatibility` (line 129) reads both out of that same cookie:

```python
first_user_info  = session.get('first_user_info', {})
second_user_info = session.get('second_user_info', {})
```

The share link in `dashboard()` is `f"{BASE_URL}/second/user/login"` — a bare URL with
no identifier in it. When B opens it on their own phone, Flask issues B a **fresh
session**. B's `first_user_info` is empty, so B's results page renders with half the
data, and A's browser never learns that B participated at all.

**This works today only if two people share one browser.** It is not a bug in an
otherwise-correct design; the design has no representation of "a match between two
people". That is the thing being added.

**The shared state moves from the cookie to the server.** The share link stops being a
bare route and becomes `/{match_id}/join`, where `match_id` is the server-side record
both participants attach themselves to.

### The flow as HTTP

| # | Request | Server does | State after |
|---|---|---|---|
| 1 | `GET /` | Render landing | — |
| 2 | `GET /login` | Create the `Match` row, then mint a signed `state` carrying `(nonce, match_id, role=inviter)`; redirect to Spotify | DB row; `session['oauth_nonce']` |
| 3 | `GET /callback?code&state` | Verify the state's signature and nonce; exchange code; fetch profile + library; **discard tokens**; `Participant(role='inviter')` | DB row; `session['role:<match_id>']` |
| 4 | `GET /m/<match_id>` | A's page: share link, "waiting for them" | — |
| 5 | `GET /m/<match_id>` | B, who is not in the match yet, gets the invite page instead: "Akin wants to compare…", what will be read, a button | — |
| 6 | `GET /m/<match_id>/join` | Mint a signed `state` carrying `(nonce, match_id, role=invited)`; redirect to Spotify | `session['oauth_nonce']` |
| 7 | `GET /callback?code&state` | Same route as step 3; the role comes out of the verified state. Attach `Participant(role='invited')`, score, store the result | DB rows; `session['role:<match_id>']` |
| 8 | `GET /m/<match_id>` | Both see the result | — |

One `/callback` route, not two. The old `/redirect-user` and `/redirect-second-user`
differed only in which session keys they wrote, which is exactly the distinction that
disappears once a `Match` exists. **This halves the OAuth code and means one redirect
URI to register with Spotify.**

**The match id and role travel in `state`, not in the session.** The original plan put
`pending_match` in B's cookie; that was changed during implementation because the
cookie is the one thing that is *not* shared across the two devices, and because
`state` already has to exist for CSRF. It is HMAC-SHA256 signed with `SECRET_KEY` and
verified in constant time, so a tampered match id or role is rejected before anything
in it is read. See `oauth_state.py`. The session's remaining job is one key per match:
`role:<match_id>`, which is how `/m/<id>` knows whether the viewer is a participant.

Note also that `/m/<match_id>` serves three audiences — inviter, invitee, bystander —
rather than the invite living at its own URL. One link to send, which is the point.

A does not poll. `/m/<id>` renders "waiting" and asks the viewer to refresh. The
plan here was a 5s `<meta http-equiv="refresh">`; the shipped template has the text
and not the tag, so this is a real (small) gap rather than a decision. See §9.

---

## 2. Data model

SQLAlchemy over SQLite locally, Postgres in deployment. Three tables.

### `matches`

| column | type | notes |
|---|---|---|
| `id` | `TEXT` PK | `secrets.token_urlsafe(16)`. **Not** a sequential int and not `uuid4().hex` in a URL you expect to be unguessable-by-inspection — 128 bits from `secrets`, which is what makes the share link safe to send over SMS |
| `status` | `TEXT` | one of six — see `MatchStatus` in `models.py` |
| `created_at` | `TIMESTAMP` | |
| `completed_at` | `TIMESTAMP` NULL | |
| `result_json` | `TEXT` NULL | the `compatibility()` dict, serialised |

Index: `status`, `created_at` (for the expiry sweep).

### `participants`

| column | type | notes |
|---|---|---|
| `id` | `INTEGER` PK | |
| `match_id` | `TEXT` FK → `matches.id` | |
| `role` | `TEXT` | `inviter` \| `invitee` |
| `spotify_user_id` | `TEXT` | from `/v1/me`, for display and dedup |
| `display_name` | `TEXT` | |
| `avatar_url` | `TEXT` NULL | |
| `tracks_json` | `TEXT` | the derived track rows — see below |
| `n_tracks` | `INTEGER` | denormalised for the guard, avoids parsing JSON to check a count |
| `coverage` | `REAL` | fraction of tracks the feature store resolved |
| `created_at` | `TIMESTAMP` | |

Unique: `(match_id, role)` — a match has at most one inviter and one invitee.
Index: `match_id`.

**`tracks_json` holds exactly what `CompatibilityScorer` consumes**, no more:

```
track_id, artist, genre, popularity, year,
danceability, energy, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, loudness, duration_ms
```

Ten audio features (`schema.AUDIO_FEATURES` = 7 bounded + 3 unbounded) plus the four
metadata columns the non-audio components need: `genre_overlap` reads `genre`,
`artist_jaccard` reads `artist`, `era_overlap` reads `year`, `obscurity_match` reads
`popularity`. Store those and nothing else. Track *names* are not needed for scoring —
include them only if the results page shows "songs you both love", which it should.

### `track_features`

The feature store, moved out of a 135 MB CSV and into the database.

| column | type |
|---|---|
| `track_id` | `TEXT` PK |
| `danceability` … `duration_ms` | `REAL` (10 columns) |

1,419,043 rows. Primary key on `track_id` makes a lookup a B-tree probe. The
alternative — `pd.read_csv` at startup — costs roughly 200 MB of resident memory **per
worker process**, so four Gunicorn workers would want 800 MB to answer a query SQLite
answers from disk in microseconds. Loaded once by a seed script, not at boot.

### Lifecycle

- A match created and never joined sits at `waiting`. A daily sweep marks anything
  older than **7 days** `expired` and nulls `tracks_json` on its participants. Nobody
  needs week-old top-track lists, and holding them is a liability for no benefit.
- The full set is `waiting`, `complete`, `insufficient`, `unresolved`, `expired`,
  `failed` — six, not three. `insufficient` and `unresolved` are **outcomes, not
  errors**: the pair did get an answer, and it was "we will not put a number on this".
  They are the match-level mirror of `compatibility()`'s two non-`ok` statuses
  (§ below), and a template that lumps them in with `failed` throws away the
  distinction the scorer went to some trouble to make. `MatchStatus` in `models.py`
  is the authority; this list is a convenience.
- **Results are computed once, at completion, and stored** in `result_json`. They are
  not recomputed on page load. Both participants must see the same number, and
  recomputing against a retrained model would silently change it under them. The stored
  result is the record of what the pair was told.

---

## 3. The OAuth `state` parameter

This is the subtlest part of the design and the easiest place to accidentally remove
the CSRF protection the current code has.

Today `state` carries a CSRF nonce and nothing else (`lovify.py:41`), checked at
`:144`. The new flow also needs the callback to know *which match* B is joining.

**Do not put the match id in `state`.** The tempting move is
`state = f"{nonce}:{match_id}"`, and it is wrong in a way that is easy to miss: `state`
comes back from the browser via a redirect an attacker can construct. If the callback
trusts a match id read out of `state`, an attacker can send a victim a link that
attaches the victim's freshly-authorised Spotify data to **the attacker's** match. The
victim signs into their own account, sees a plausible Lovify page, and the attacker
reads their top tracks. `state` is attacker-influenced data; it authenticates the
*round trip*, not the *intent*.

**Keep `state` as a pure random nonce, and carry the match id in the server-side
session.**

```
GET /login?m=<match_id>
    session['oauth_state']   = secrets.token_urlsafe(32)
    session['pending_match'] = match_id        # validated: exists, status=pending
    -> redirect to Spotify with state=<oauth_state>

GET /callback?code&state
    reject unless state == session.pop('oauth_state')
    match_id = session.pop('pending_match', None)   # server-side, never from the URL
```

The match id then travels in a cookie the attacker cannot write, and `state` keeps
doing the one job it is good at. Both values are `pop`-ed, so a replayed callback
fails.

Also: `session.pop`, not `session.get`. The current code leaves `session['uuid']` in
place after a successful callback, so the same `code`/`state` pair replays cleanly.

---

## 4. Security

**Spotify tokens are never persisted.** They exist inside the `/callback` request and
are discarded before it returns. The callback does, in order: exchange code → fetch
`/v1/me` → fetch top tracks → attach genres via `/v1/artists` → join the feature store
→ write `tracks_json` → drop the token.

This is the single highest-leverage decision in the document. It means:

- A database compromise leaks display names and track lists, not account access.
- There is no refresh-token rotation to implement, no expiry to handle, no encryption
  at rest to get wrong, and no revocation story to build.
- Re-reading someone's library later is impossible without asking them again — which
  is the correct default for a thing that compares two people once.

The cost is real and worth stating: a user who wants an updated match must re-authorise.
For this product that is a feature.

**The session cookie keeps a reduced job**: the OAuth nonce, the pending match id, and
`participant_id` so a returning browser can be shown "your match". It no longer carries
tokens or track data. Flask's signed cookies cap around 4 KB, which the old design was
heading toward with two users' tokens and profiles in it.

**What an attacker holding a share link can do**: open the join page, see the inviter's
display name and avatar, and join the match themselves — at which point they see the
compatibility result and the inviter's shared artists. That is the same power the
intended recipient has, which is inherent to "a link you text to someone". It is
bounded: 128 bits of entropy makes guessing infeasible, a match accepts exactly one
invitee, and `complete` matches reject further joins. They cannot read the inviter's
full track list, their email, or anything about their Spotify account beyond a public
display name.

**What they cannot do**: act on the inviter's Spotify account, since no token exists to
steal.

---

## 5. The 25-user cap

Spotify development mode allows **25 users total**, each added by hand in the developer
dashboard by the exact email on their Spotify account. Anyone not on that list gets an
authorisation error they cannot resolve and will read as "this app is broken".

**Both participants in a match must be allowlisted.** A demo where the audience scans a
QR code and tries it does not work. Extended quota mode requires Spotify review, is not
granted for student projects in practice, and would not restore the deprecated
endpoints anyway.

Plan for demo day accordingly: collect the Spotify account emails of everyone who might
demo, in advance, and add them. Twelve allowlisted people is six simultaneous matches.
Treat this as a hard constraint on the product, not an operational footnote — it shapes
what a launch can even mean.

---

## 6. Where the ML sits

`ml/` is a **library the app imports**, not a service it calls. No HTTP between them, no
separate deployment, no serialisation boundary to maintain.

```python
# app startup, once
from lovify_ml.service import LovifyModels
MODELS = LovifyModels.load(os.environ["MODEL_PATH"])
```

**Load at startup, never per request.** The artifact carries the catalog and the fitted
null ladder; constructing it per request would add seconds to every page.

`compatibility()` returns a dict whose `status` is one of three values, and **the UI
must render all three**. The two non-`ok` ones are deliberate product states, not
errors, and must not become a 500 or a generic "something went wrong":

| `status` | Means | UI must show |
|---|---|---|
| `ok` | scored | percentile, components, shared artists |
| `insufficient_library` | fewer than `MIN_LIBRARY_TRACKS` (100) usable tracks | "not enough listening history to score this fairly yet" — and the shared artists, which are exact regardless |
| `no_resolvable_tracks` | zero tracks survived the feature-store join | "we couldn't resolve any of their music — usually too new or too obscure" |

The refusals exist because the alternative is a confidently wrong number: scoring a
thin library against a full-size reference population understates it by tens of
percentile points in a direction the user cannot detect. Rendering them as errors would
throw away the entire point.

`taste_profile()` similarly returns `is_distinctive: False` when a user's top sign is
not meaningfully above the population baseline. The UI should soften its language in
that case rather than asserting a personality.

---

## 7. The risk this plan does not yet solve

**The expected number of usable tracks per user may fall below the threshold that
allows scoring at all.** Worked through:

- `/v1/me/top/tracks` returns at most **50** per call. Three time ranges
  (`short_term`, `medium_term`, `long_term`) give at most 150, and they overlap
  heavily — realistically **~100–120 unique**.
- Feature-store coverage measured on real scraped tracks is **41.1%**.
- 110 × 0.41 ≈ **45 tracks with audio features**.
- `MIN_LIBRARY_TRACKS = 100`.

**So the common case may be `insufficient_library` for both participants.** The product
would politely refuse to work for most pairs, which is worse than being wrong.

This needs resolving before the UI is built on the assumption that scoring succeeds.
Three options, none free:

1. **Add `/v1/me/tracks`** (saved/liked songs, paginated — `spotify.py` already has
   `saved_tracks(max_items=200)`). More volume, but liked songs are a different signal
   from top tracks and mixing them changes what is being compared.
2. **Lower `MIN_LIBRARY_TRACKS`.** It was set to 100 deliberately, over a defensible 50,
   because the measurements came from synthetic users. Lowering it to 50 is arguably
   fine *and* should be justified by measurement on real libraries, not by the fact that
   it is inconvenient.
3. **Score on non-audio components when coverage is low.** Three of the five components
   — genre, artist, era, obscurity — need no audio features at all. A degraded-but-real
   score beats a refusal, if it is labelled as degraded.

**Do not pick one of these in code without measuring first.** The first real pair of
allowlisted users gives the actual number; until then this is arithmetic, not evidence.

---

## 8. Migration path

Reviewable steps, each shippable. Not one rewrite.

| # | Step | Touches |
|---|---|---|
| 0 | Merge PR #1 and PR #2 | — |
| 1 | Add SQLAlchemy, models, migrations; seed `track_features` from `feature_store.csv` | new `app/models.py`, `scripts/seed_features.py` |
| 2 | Collapse `/redirect-user` + `/redirect-second-user` into one `/callback`; keep existing session behaviour | `lovify.py` |
| 3 | Introduce `Match`/`Participant`; write to DB alongside the session; nothing reads the DB yet | `lovify.py`, `app/models.py` |
| 4 | Move the fetch-and-derive work into the callback; stop storing tokens | `lovify.py`, uses `ml/spotify.py` |
| 5 | Switch `/m/<id>` and the join page to read from the DB; delete the session reads | templates + routes |
| 6 | Wire `LovifyModels` in; render all three `status` values | `lovify.py`, `compatibilityV2.html` |
| 7 | Delete `first_user_*` / `second_user_*` session keys entirely | `lovify.py` |

**Kept**: the Jinja templates and CSS (`index.html`, `user_index.html`,
`compatibilityV2.html`, `about.html` and their stylesheets), the OAuth exchange
helpers, the overall visual design. This is a state-management rewrite, not a visual one.

**Replaced**: every `session[...]` read of participant data, the two callback routes,
the bare share link.

**Deleted**: `loading.html` and `compatibility.html` if `compatibilityV2.html` is the
live one — dead templates are a trap for whoever edits next.

---

## 9. Decisions, and what we are not doing yet

**Monorepo, one deployable.** Six people and one app. A separate frontend repo buys
version skew between two things that always ship together, plus a second CI pipeline,
and returns nothing. Revisit if the frontend is ever consumed by something other than
this backend.

**Server-rendered Flask. No SPA migration.** This is the decision most likely to be
second-guessed, so: the app is three pages joined by OAuth redirects, which is the
exact shape server rendering handles best. A React or Next front end would add a build
step, a CORS surface, client-side token handling, and a second deploy target — in
exchange for no behaviour the user can perceive. The templates already exist and work.
Revisit if the UI becomes genuinely interactive (live-updating match feed, in-page
playback).

**No Redis.** The only shared state is in Postgres and the only background job is a
daily expiry sweep, which a cron-style task handles. Add Redis when there is a real
queue — sending email, or fanning out to many participants.

**No Celery / task queue.** The fetch-and-derive work runs inside the callback. It is
2–4 Spotify calls and a database join, which is comfortably inside a request. If it
ever exceeds ~5 seconds, move it to a job and show a progress page — that is also when
`loading.html` stops being dead.

**No polling endpoint / websockets.** A meta-refresh on the waiting page is enough for
two people who are already texting. Replace it when matches involve more than two
people or the wait becomes long enough to notice.

**No user accounts.** There is no login beyond Spotify OAuth, no password, no profile.
A participant is a row attached to a match. Add accounts when someone needs to see
their match history — and note that doing so changes the data-retention story, because
then you are keeping track lists on purpose.

**Docker Compose for local development.** `docker compose up` gives Flask + Postgres
with the feature store seeded. Railway deploys from the same Dockerfile, so "works on
my machine" and "works in production" stop diverging. The cost is that everyone needs
Docker installed; the alternative is six people hand-rolling Postgres.
