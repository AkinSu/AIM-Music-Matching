#!/usr/bin/env python3
"""Run the whole match flow locally, with no Spotify account involved.

Why this exists as a separate script rather than a DEMO_MODE flag: a flag
that swaps out authentication is one environment variable away from being
set in production by accident. There is no such flag. This file imports the
app and replaces its Spotify client afterwards, so the only way to get the
fake one is to run this file deliberately, and production never does.

What you get:

    http://127.0.0.1:8080/              the real landing page
    http://127.0.0.1:8080/demo          two fake sign-in links, side by side

Click "Sign in as Ada", copy the invite link, open it in a *private window*,
and sign in as Bo. Both of you then see a real score, computed by the real
code, from a small synthetic feature store.

    python scripts/demo.py
"""

from __future__ import annotations

import os
import secrets
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "LOVIFY ME APP"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "tests"))

# Must be set before `import lovify`: the module reads them at import time.
os.environ["SECRET_KEY"] = secrets.token_hex(32)
os.environ["BASE_URL"] = "http://127.0.0.1:8080"
os.environ["DATABASE_URL"] = "sqlite:///" + str(
    Path(tempfile.mkdtemp(prefix="lovify-demo-")) / "demo.db")
os.environ.setdefault("SPOTIFY_CLIENT_ID", "demo")
os.environ.setdefault("SPOTIFY_CLIENT_SECRET", "demo")

import lovify  # noqa: E402
import models  # noqa: E402
from conftest import StubScorer, give_library, seed_features, track_ids  # noqa: E402
from flask import render_template_string  # noqa: E402
from spotify_client import FakeSpotifyClient  # noqa: E402

# Refuse to run anywhere that is not a developer's own machine. The fake
# client accepts any login, so this must never be reachable from a network.
if not lovify.BASE_URL_IS_LOCAL:          # pragma: no cover
    sys.exit(f"demo.py refuses to run with BASE_URL={lovify.BASE_URL}. "
             "It replaces authentication with a stub.")

PEOPLE = {
    "ada": ("Ada", track_ids("shared", 90) + track_ids("ada", 70)),
    "bo": ("Bo", track_ids("shared", 90) + track_ids("bo", 70)),
    "cy": ("Cy", track_ids("cy", 160)),          # very different taste
    "dot": ("Dot", track_ids("shared", 20)),     # too small to score
}

DEMO_PAGE = """
<!doctype html><meta charset=utf-8><title>Lovify demo</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem;line-height:1.6}
 a.btn{display:inline-block;padding:.5rem .9rem;margin:.2rem .3rem .2rem 0;
       border:1px solid #888;border-radius:8px;text-decoration:none;color:inherit}
 code{background:#eee;padding:.1rem .3rem;border-radius:4px}
 ol{padding-left:1.2rem}
</style>
<h1>Lovify demo</h1>
<p>No Spotify account involved. These four people have synthetic libraries,
   and the score you get is computed by the real code.</p>

<h2>Start a match as…</h2>
{% for code, (name, tracks) in people.items() %}
  <a class=btn href="/login?demo={{ code }}">Sign in as {{ name }}
     <small>({{ tracks|length }} tracks)</small></a>
{% endfor %}

<h2>What to try</h2>
<ol>
  <li>Sign in as <b>Ada</b>. You land on a waiting page with an invite link.</li>
  <li>Copy that link and open it in a <b>private window</b> — that is a
      second device. Sign in as <b>Bo</b>.</li>
  <li>Both windows now show a score. Ada and Bo share 90 tracks, so it is high.</li>
  <li>On the result page, press <b>Create a public link</b>. Open the
      <code>/s/…</code> link it gives you in a <b>third</b> window.</li>
  <li>In that third window, try the original invite link. You should get
      <b>403</b> — results are for participants only.</li>
  <li>Press <b>Replace with a new link</b>. The first <code>/s/…</code> link
      should now 404.</li>
</ol>

<h2>Other things worth seeing</h2>
<ul>
  <li><b>Cy</b> shares nothing with Ada — a low score, and no genre chips.</li>
  <li><b>Dot</b> has 20 tracks, under the 100 the scorer requires. You get
      "not enough listening history", which is a deliberate refusal rather
      than an error.</li>
  <li><code>/me/matches</code> — your history, once you have one.</li>
</ul>
"""


START_PAGE = """
<!doctype html><meta charset=utf-8><title>Sign in — Lovify demo</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;line-height:1.6}
 a.btn{display:inline-block;padding:.5rem .9rem;margin:.2rem .3rem .2rem 0;
       border:1px solid #888;border-radius:8px;text-decoration:none;color:inherit}
 p.note{color:#666;font-size:.95rem}
</style>
<h1>Sign in as…</h1>
<p class=note>This is the demo. There is no real Spotify behind it, so pick
   one of these four instead. Start with <b>Ada</b>.</p>
{% for code, (name, tracks) in people.items() %}
  <a class=btn href="/login?demo={{ code }}">{{ name }}
     <small>({{ tracks|length }} tracks)</small></a>
{% endfor %}
<p class=note><a href="/demo">← what to try</a></p>
"""


JOIN_PAGE = """
<!doctype html><meta charset=utf-8><title>Join this match</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;line-height:1.6}
 a.btn{display:inline-block;padding:.5rem .9rem;margin:.2rem .3rem .2rem 0;
       border:1px solid #888;border-radius:8px;text-decoration:none;color:inherit}
</style>
<h1>Join as…</h1>
<p>In the real app this button goes to Spotify. Pick a demo person instead —
   it has to be someone who is not already in this match.</p>
{% for code, (name, tracks) in people.items() %}
  <a class=btn href="/m/{{ match_id }}/join?demo={{ code }}">{{ name }}
     <small>({{ tracks|length }} tracks)</small></a>
{% endfor %}
"""


def build() -> None:
    models.Base.metadata.drop_all(lovify.engine)
    models.init_db(lovify.engine)

    spotify = FakeSpotifyClient()
    everything: set[str] = set()
    for code, (name, tracks) in PEOPLE.items():
        give_library(spotify, code, name, top=tracks)
        everything |= set(tracks)

    with lovify.Session() as db:
        seed_features(db, sorted(everything))

    lovify.app.spotify = spotify
    # The real scorer needs a joblib artifact that is gitignored. The stub
    # keeps the contract -- including the refusal below 100 tracks, which is
    # the behaviour most worth seeing.
    lovify.app.scorer = StubScorer()

    # Flask refuses to add a route to an app that has already served a
    # request, because in a normal app that means a route some workers have
    # and others do not. Here the mutation is the entire point of the file
    # and it happens in one process, so clear the flag and put it back.
    # (It matters in tests, which reuse one app object across cases.)
    was_served = lovify.app._got_first_request
    lovify.app._got_first_request = False
    try:
        if "demo_index" not in lovify.app.view_functions:
            lovify.app.add_url_rule(
                "/demo", "demo_index",
                lambda: render_template_string(DEMO_PAGE, people=PEOPLE))
    finally:
        lovify.app._got_first_request = was_served

    # /login?demo=<code> short-circuits the Spotify round trip: mint the
    # state the app would have minted, then call its own callback with it.
    real_login = lovify.app.view_functions["login_redirect"]
    if getattr(real_login, "_demo_wrapped", False):
        return                      # build() already ran in this process

    def login(*a, **kw):
        """Every route into the app lands here, not just the /demo page.

        The landing page's own "Log in with Spotify" button goes to /login
        with no demo code, and the first version of this file passed that
        straight through to the real OAuth redirect -- which, with the fake
        client installed, is a hostname that does not resolve. Anyone who
        started from the front page instead of /demo got "Safari can't find
        the server spotify.test". Show the chooser instead.
        """
        from urllib.parse import parse_qs, urlparse

        from flask import redirect, request
        code = request.args.get("demo")
        if not code:
            return render_template_string(START_PAGE, people=PEOPLE)
        started = real_login()
        state = parse_qs(urlparse(started.headers["Location"]).query)["state"][0]
        return redirect(f"/callback?code={code}&state={state}")

    login._demo_wrapped = True
    lovify.app.view_functions["login_redirect"] = login

    real_join = lovify.app.view_functions["join_match"]

    def join(match_id):
        """The invite page's button lands here.

        Without a `demo` code it would bounce to the real Spotify, which is
        not available. Show a chooser instead -- which is what the second
        device's sign-in step actually is, in a demo.
        """
        from urllib.parse import parse_qs, urlparse

        from flask import redirect, request
        code = request.args.get("demo")
        if not code:
            return render_template_string(
                JOIN_PAGE, people=PEOPLE, match_id=match_id)
        started = real_join(match_id)
        if getattr(started, "status_code", None) != 302:
            return started          # expired, already full, or an error page
        state = parse_qs(urlparse(started.headers["Location"]).query)["state"][0]
        return redirect(f"/callback?code={code}&state={state}")

    join._demo_wrapped = True
    lovify.app.view_functions["join_match"] = join


def main() -> None:
    build()
    print("\n  Lovify demo")
    print("  ───────────")
    print("  open  http://127.0.0.1:8080/demo")
    print("  people: " + ", ".join(n for n, _ in PEOPLE.values()))
    print("  database: " + os.environ["DATABASE_URL"] + "  (temporary)\n")
    lovify.app.run(host="127.0.0.1", port=8080, debug=False)


if __name__ == "__main__":
    main()
