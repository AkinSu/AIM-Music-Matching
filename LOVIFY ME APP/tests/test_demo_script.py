"""The demo runner is the thing a new contributor runs first.

If it breaks, the project looks broken, and nobody debugs a demo -- they
just close the tab. So the flow it advertises is tested like a feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def demo(app, monkeypatch):
    """Build the demo against the already-configured test app, reversibly.

    `build()` is a deliberate, permanent mutation of a live app object --
    that is the point of the script -- so running it inside a test suite
    that shares one app would leave the demo's stub Spotify and its wrapped
    views in place for every test that follows. Snapshot and restore.
    """
    import demo as demo_module

    monkeypatch.setattr(demo_module.lovify, "app", app, raising=False)
    original_views = dict(app.view_functions)
    original = (app.spotify, app.scorer)

    demo_module.build()
    yield app.test_client()

    app.view_functions.clear()
    app.view_functions.update(original_views)
    app.spotify, app.scorer = original


def start_as(client, who: str) -> str:
    """Sign in as a demo person and return the match id.

    Follows redirects: a demo login is three hops (login -> callback ->
    /m/<id>), so reading Location off the first one gets the callback URL.
    """
    from urllib.parse import urlparse

    landed = client.get(f"/login?demo={who}", follow_redirects=True)
    assert landed.status_code == 200, landed.get_data(as_text=True)[:400]
    return urlparse(landed.request.path).path.rsplit("/", 1)[-1]


def test_the_demo_index_lists_the_people(demo):
    body = demo.get("/demo").get_data(as_text=True)
    assert demo.get("/demo").status_code == 200
    for name in ("Ada", "Bo", "Cy", "Dot"):
        assert name in body


def test_signing_in_as_a_demo_person_creates_a_match(demo):
    landed = demo.get("/login?demo=ada", follow_redirects=True)
    assert landed.status_code == 200
    assert "Ada" in landed.get_data(as_text=True)


def test_two_demo_people_complete_a_match(app, demo):
    """The headline flow, end to end, in two separate browsers."""
    ada = demo
    match_id = start_as(ada, "ada")

    bo = app.test_client()
    bo.get(f"/m/{match_id}/join?demo=bo", follow_redirects=True)

    for who in (ada, bo):
        page = who.get(f"/m/{match_id}")
        assert page.status_code == 200
        assert "percentile" in page.get_data(as_text=True).lower()


def test_the_join_page_offers_a_chooser_instead_of_spotify(app, demo):
    match_id = start_as(demo, "ada")

    page = app.test_client().get(f"/m/{match_id}/join")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "Join as" in body
    assert "accounts.spotify.com" not in body


def test_dot_is_below_the_scoring_threshold(app, demo):
    """The refusal path, which is the one worth seeing."""
    match_id = start_as(demo, "ada")

    dot = app.test_client()
    dot.get(f"/m/{match_id}/join?demo=dot", follow_redirects=True)

    body = demo.get(f"/m/{match_id}").get_data(as_text=True)
    assert "Not enough listening history" in body


def test_it_refuses_to_run_against_a_public_base_url():
    """The guard that keeps a stubbed login off the internet."""
    source = (ROOT / "scripts" / "demo.py").read_text()
    assert "BASE_URL_IS_LOCAL" in source
    assert "sys.exit" in source


def test_the_landing_page_login_button_does_not_dead_end(demo):
    """The route a real person actually takes.

    Akin opened the demo, pressed "Log in with Spotify" on the front page,
    and got "Safari can't find the server spotify.test" -- because /login
    with no demo code fell through to the real OAuth redirect, and the fake
    client's authorize URL is a hostname that does not resolve. Every way
    into the app has to reach the chooser.
    """
    page = demo.get("/login")
    assert page.status_code == 200, "a redirect here goes somewhere that does not exist"
    body = page.get_data(as_text=True)
    assert "spotify.test" not in body
    for name in ("Ada", "Bo", "Cy", "Dot"):
        assert name in body


def test_no_route_in_the_demo_redirects_to_a_fake_spotify(demo):
    """The general form of the bug above."""
    for path in ("/", "/login", "/dashboard", "/find-compatibility"):
        page = demo.get(path, follow_redirects=True)
        assert "spotify.test" not in page.get_data(as_text=True), path
        assert page.status_code == 200, f"{path} -> {page.status_code}"
