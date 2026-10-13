"""The app must refuse to start without a SECRET_KEY.

Carried forward from the security fix on the SECRET_KEY branch. The failure
mode it prevents is not a crash but the opposite: silently falling back to a
literal key that is committed in this repository's public history, which
would let anyone forge a session cookie -- and now also an OAuth state,
since both are signed with this key.

Run in a subprocess because the check happens at import, and `lovify` is
already imported by the rest of the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]


# A developer tree has a real .env at the repository root, and load_dotenv()
# walks up to find it -- so an empty `env=` is not an empty environment. The
# stub neutralises that one call; everything else about the import is real.
IMPORT_WITHOUT_DOTENV = (
    "import sys, types;"
    "m = types.ModuleType('dotenv'); m.load_dotenv = lambda *a, **k: None;"
    "sys.modules['dotenv'] = m;"
    "import lovify"
)


def _import_lovify_with(env_overrides: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
    env.update(PYTHONPATH=str(APP_DIR), **env_overrides)
    return subprocess.run([sys.executable, "-c", IMPORT_WITHOUT_DOTENV],
                          capture_output=True, text=True, env=env, cwd=APP_DIR)


def test_no_secret_key_is_a_refusal_to_start_not_a_fallback():
    done = _import_lovify_with({})

    assert done.returncode != 0
    assert "SECRET_KEY is not set" in done.stderr
    # The message has to tell someone what to do about it.
    assert "secrets.token_hex" in done.stderr


def test_with_a_secret_key_the_app_imports_cleanly(tmp_path):
    done = _import_lovify_with({
        "SECRET_KEY": "a-key",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'boot.db'}",
    })

    assert done.returncode == 0, done.stderr


# ── missing credentials ──────────────────────────────────────────────────────

class TestNoSpotifyCredentials:
    """Akin filled in .env, pressed log in, and got a blank Spotify page
    reading "client_id: Not present".

    The app had built `…/authorize?client_id=&…` and sent him there. Spotify's
    error names neither this app nor .env, so the only available conclusion
    was that the app was broken. It was not: the container had been started
    before .env was filled in, and compose reads that file once, at startup.

    The app must say so itself rather than delegating the explanation to
    Spotify.
    """

    @pytest.fixture
    def unconfigured(self, app):
        from spotify_client import SpotifyClient

        original = app.spotify
        app.spotify = SpotifyClient(client_id="", client_secret="",
                                    redirect_uri="http://127.0.0.1:8080/callback",
                                    scope="user-top-read")
        yield app.test_client()
        app.spotify = original

    def test_the_client_knows_it_is_not_configured(self, unconfigured, app):
        assert app.spotify.configured is False

    def test_it_refuses_to_build_a_url_with_an_empty_client_id(self, app, unconfigured):
        from spotify_client import SpotifyNotConfigured

        with pytest.raises(SpotifyNotConfigured):
            app.spotify.authorize_url("some-state")

    def test_login_explains_itself_instead_of_redirecting(self, unconfigured):
        page = unconfigured.get("/login")
        assert page.status_code == 503, "a redirect here goes to a blank Spotify error"
        body = page.get_data(as_text=True)
        assert "SPOTIFY_CLIENT_ID" in body
        assert "restart" in body.lower(), "the actual fix is usually a restart"

    def test_a_configured_client_still_redirects(self, app):
        """The suite runs against the fake, so assert the property that
        matters rather than the hostname: a redirect happens, and it does
        not carry an empty client id."""
        started = app.test_client().get("/login")
        assert started.status_code == 302
        assert "client_id=&" not in started.headers["Location"]

    def test_the_fake_client_honours_the_same_contract(self, app):
        """A double that cannot be unconfigured cannot catch this bug."""
        from spotify_client import SpotifyNotConfigured

        assert app.spotify.configured is True
        app.spotify.configured = False
        try:
            with pytest.raises(SpotifyNotConfigured):
                app.spotify.authorize_url("s")
            assert app.test_client().get("/login").status_code == 503
        finally:
            app.spotify.configured = True
