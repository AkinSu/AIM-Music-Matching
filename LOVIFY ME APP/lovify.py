"""Lovify: compare two people's music taste across two devices.

What changed and why
--------------------
The previous version kept both participants in `session[...]` -- one browser's
cookie -- and the share link pointed at `/second/user/login`. That silently
assumed the second person was sitting in the first person's browser. A real
invitee on their own phone got a fresh session with no trace of the inviter,
so the results page had half the data and no way to find the rest. The
headline feature could not work across devices at all.

Shared state now lives in a `Match` row keyed by an unguessable id, and that
id travels in the share link and through OAuth. The cookie session is reduced
to one job: remembering which participant *this browser* is.

Spotify tokens are used once, during the callback, to read the person's top
tracks, and are then discarded. Nothing long-lived is stored, so there is no
refresh logic and nothing worth stealing.
"""

from __future__ import annotations

import os
from urllib.parse import urljoin

from dotenv import load_dotenv
from flask import (
    Flask, redirect, render_template, request, session, url_for,
)

import matching
import oauth_state
from models import MatchStatus, Role, init_db, make_engine, make_session_factory
from scoring import Scorer
from spotify_client import SpotifyClient, SpotifyError

load_dotenv()

app = Flask(__name__, template_folder="templates", static_url_path="/static")

# BASE_URL: the public origin. Must match the redirect URI registered with
# Spotify exactly, including scheme and any trailing path.
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080").rstrip("/")
BASE_URL_IS_LOCAL = BASE_URL.startswith(("http://127.0.0.1", "http://localhost"))
REDIRECT_URI = f"{BASE_URL}/callback"

# Flask signs session cookies with this key, and the session identifies which
# participant a browser is. A missing key must be a crash, never a silent
# downgrade to a literal that is committed in this repository's public
# history. (Carried forward from the security fix on the SECRET_KEY branch;
# this branch was cut from main, which does not have it yet.)
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError(
        "SECRET_KEY is not set. Generate one with:\n"
        "    python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "then put it in .env for local development, or in the hosting "
        "provider's environment variables for a deployment. Do not commit it."
    )

app.config["SESSION_COOKIE_NAME"] = "lovify_session"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",   # still permits the OAuth redirect back
    SESSION_COOKIE_SECURE=not BASE_URL_IS_LOCAL,
)

# ── wiring, all overridable in tests ─────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///lovify.db")
engine = make_engine(DATABASE_URL)
init_db(engine)
Session = make_session_factory(engine)

# Loaded once, at import, rather than per request.
app.scorer = Scorer()
if not app.scorer.ready:
    app.logger.warning("Compatibility scoring disabled: %s",
                       app.scorer.unavailable.reason)

app.spotify = SpotifyClient(
    client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
    redirect_uri=REDIRECT_URI,
    scope=os.getenv("SCOPE", "user-top-read"),
)

# How many track ids to gather per person before we stop asking Spotify.
# 400 is chosen against measured coverage: resolution against the feature
# store runs 41% overall and ~60% for the popular tracks that dominate a
# personal library, so 400 fetched yields roughly 165-240 usable -- comfortably
# past MIN_LIBRARY_TRACKS (100) with room for a light listener.
#
# Do not "fix" a shortfall by lowering MIN_LIBRARY_TRACKS: that threshold is
# 100 because below it the compatibility percentile is biased by tens of
# points, which is a property of the scoring, not of the fetch.
LIBRARY_BUDGET = int(os.getenv("LIBRARY_BUDGET", "400"))


def db_session():
    return Session()


def _start_oauth(match_id: str, role: str):
    """Mint a signed state, remember its nonce, and hand off to Spotify."""
    nonce = oauth_state.new_nonce()
    session[oauth_state.SESSION_NONCE_KEY] = nonce
    state = oauth_state.build(app.secret_key, nonce, match_id, role)
    return redirect(app.spotify.authorize_url(state))


def _remember_participation(match_id: str, role: str) -> None:
    """The session's whole job: which participant is this browser."""
    session[f"role:{match_id}"] = role


def _role_in(match_id: str) -> str | None:
    return session.get(f"role:{match_id}")


def share_link(match_id: str) -> str:
    return urljoin(BASE_URL + "/", f"m/{match_id}")


# ── pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/what-about-us")
def about_page():
    return render_template("about.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home_page"))


@app.route("/login")
def login_redirect():
    """Start a new match as the inviter."""
    with db_session() as db:
        match = matching.create_match(db)
        match_id = match.id
    return _start_oauth(match_id, Role.INVITER)


@app.route("/m/<match_id>")
def match_page(match_id: str):
    """The share link.

    Three audiences land here: the inviter checking progress, the invitee
    arriving from a text message, and anyone else with the link. What they see
    depends on the match state and whether they are already in it.
    """
    with db_session() as db:
        try:
            match = matching.get_match(db, match_id)
        except matching.MatchError as exc:
            return render_template("match_error.html", message=str(exc)), 404

        role = _role_in(match_id)
        vm = matching.view_model(match, role)

        if role is None and match.invited is None:
            # A newcomer looking at an open invitation.
            return render_template(
                "match_invite.html",
                inviter_name=(match.inviter.display_name if match.inviter else "Someone"),
                match_id=match_id,
            )

        if match.status == MatchStatus.WAITING:
            return render_template("match_waiting.html",
                                   share_url=share_link(match_id), **vm)

        return render_template("match_result.html", **vm)


@app.route("/m/<match_id>/join")
def join_match(match_id: str):
    """The invitee consents and is sent to Spotify."""
    with db_session() as db:
        try:
            matching.get_match(db, match_id)
        except matching.MatchError as exc:
            return render_template("match_error.html", message=str(exc)), 404
    return _start_oauth(match_id, Role.INVITED)


# ── OAuth callback, for both roles ───────────────────────────────────────────

@app.route("/callback")
def oauth_callback():
    """One callback for both participants; `state` says which.

    The state is signed, so a tampered match id or role is rejected before
    anything in it is trusted. See oauth_state.py for the scheme and the
    attack it prevents.
    """
    if request.args.get("error"):
        return render_template(
            "match_error.html",
            message="You declined the Spotify permission, so there is nothing to compare."), 400

    nonce = session.pop(oauth_state.SESSION_NONCE_KEY, None)
    try:
        match_id, role = oauth_state.parse(app.secret_key,
                                           request.args.get("state"), nonce)
    except oauth_state.StateError as exc:
        app.logger.warning("Rejected OAuth state: %s", exc)
        return render_template(
            "match_error.html",
            message="That login could not be verified. Please start again from the link."), 400

    code = request.args.get("code")
    if not code:
        return render_template("match_error.html",
                               message="Spotify did not return a login code."), 400

    try:
        token = app.spotify.exchange_code(code)
        profile = app.spotify.profile(token)
        # Every source we have scope for, not just top tracks. Top tracks
        # alone yield ~110 unique ids across the three time ranges and only
        # ~60% of those resolve against the feature store -- about 66 usable,
        # short of the 100 the scorer requires. Saved songs are what get an
        # ordinary listener over the line.
        library = app.spotify.collect_library(token, budget=LIBRARY_BUDGET)
        track_ids, sources = library["track_ids"], library["sources"]
    except SpotifyError as exc:
        status = 403 if exc.allowlist else 502
        return render_template("match_error.html", message=str(exc)), status
    finally:
        # Explicit about the design: the token's whole life is this block.
        token = None

    app.logger.info("match=%s role=%s fetched=%d sources=%s",
                    match_id, role, len(track_ids), sources)

    with db_session() as db:
        try:
            match = matching.get_match(db, match_id)
            matching.join_match(db, match, role, profile, track_ids,
                                scorer=app.scorer, sources=sources)
        except matching.MatchError as exc:
            return render_template("match_error.html", message=str(exc)), 409

    _remember_participation(match_id, role)
    return redirect(url_for("match_page", match_id=match_id))


# ── compatibility with the old URLs ──────────────────────────────────────────
# Templates still call url_for('compatibility_page'), and old share links are
# in people's messages. Redirect rather than delete, so nothing 404s.

@app.route("/dashboard")
def dashboard():
    for key in session:
        if key.startswith("role:"):
            return redirect(url_for("match_page", match_id=key.split(":", 1)[1]))
    return redirect(url_for("login_redirect"))


@app.route("/find-compatibility")
def compatibility_page():
    return redirect(url_for("dashboard"))


@app.route("/second/user/login")
def login_two_redirect():
    """The old cross-device share link.

    It could never work -- it relied on the invitee being in the inviter's
    browser -- so anyone following an old link gets an explanation instead of
    a broken page.
    """
    return render_template(
        "match_error.html",
        message=("That is an old-style invite link, which could not work on a "
                 "second device. Ask for a new link.")), 410


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
