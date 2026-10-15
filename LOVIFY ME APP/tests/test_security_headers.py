"""The headers, asserted rather than assumed.

A security header is the kind of thing that gets added once, admired, and
then silently dropped by a refactor nobody connected to it. These tests make
that loud.
"""

from __future__ import annotations

import pytest

import security


def directives(csp: str) -> dict[str, str]:
    out = {}
    for part in csp.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(" ")
        out[name] = value.strip()
    return out


@pytest.fixture
def headers(app):
    return app.test_client().get("/").headers


# ── the smaller headers ──────────────────────────────────────────────────────

class TestHeaders:
    def test_mime_sniffing_is_off(self, headers):
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_the_page_cannot_be_framed(self, headers):
        """Clickjacking. A share card is exactly what you would frame."""
        assert headers["X-Frame-Options"] == "DENY"
        assert directives(headers["Content-Security-Policy"])["frame-ancestors"] == "'none'"

    def test_the_referrer_does_not_carry_the_url(self, headers):
        """The share URL is a bearer token and the card links out to X."""
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_hardware_permissions_are_denied(self, headers):
        policy = headers["Permissions-Policy"]
        for feature in ("geolocation", "camera", "microphone", "payment"):
            assert f"{feature}=()" in policy

    def test_hsts_is_absent_locally(self, headers):
        """An HSTS header from localhost pins 127.0.0.1 to https for a year
        in that browser, breaking every other local project the developer
        owns. It belongs in deployment only."""
        assert "Strict-Transport-Security" not in headers

    def test_hsts_is_a_year_and_preload_eligible_in_deployment(self):
        assert "max-age=31536000" in security.HSTS
        assert "includeSubDomains" in security.HSTS


# ── the content security policy ──────────────────────────────────────────────

class TestContentSecurityPolicy:
    def test_injected_script_cannot_run(self, headers):
        """The one directive that matters. No 'unsafe-inline', ever.

        If this fails, someone added an inline <script> or an onclick and
        loosened the policy to make it work. The fix is to move the code
        into static/lovify.js, not to widen this.
        """
        script_src = directives(headers["Content-Security-Policy"])["script-src"]
        assert "'unsafe-inline'" not in script_src
        assert "'unsafe-eval'" not in script_src

    def test_inline_style_is_permitted_and_that_is_deliberate(self, headers):
        """The result page's component bars set a computed width."""
        style_src = directives(headers["Content-Security-Policy"])["style-src"]
        assert "'unsafe-inline'" in style_src

    def test_everything_defaults_to_self(self, headers):
        d = directives(headers["Content-Security-Policy"])
        assert d["default-src"] == "'self'"
        assert d["object-src"] == "'none'"
        assert d["base-uri"] == "'none'"

    def test_forms_can_only_post_to_us_or_spotify(self, headers):
        assert directives(headers["Content-Security-Policy"])["form-action"] == (
            "'self' https://accounts.spotify.com")

    def test_deployment_upgrades_insecure_subresources(self):
        assert "upgrade-insecure-requests" in security.content_security_policy(local=False)
        assert "upgrade-insecure-requests" not in security.content_security_policy(local=True)

    def test_the_policy_is_on_the_public_card_too(self, app, finished_token):
        page = app.test_client().get(f"/s/{finished_token}")
        assert page.status_code == 200
        assert "Content-Security-Policy" in page.headers
        assert page.headers["X-Frame-Options"] == "DENY"


# ── cookies ──────────────────────────────────────────────────────────────────

class TestSessionCookie:
    def test_script_cannot_read_it(self, app):
        assert app.config["SESSION_COOKIE_HTTPONLY"] is True

    def test_it_does_not_ride_on_cross_site_requests(self, app):
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"

    def test_the_public_card_sets_no_cookie(self, app, finished_token):
        """Someone opening a shared link from a group chat is not a user of
        ours and should not be given a session."""
        page = app.test_client().get(f"/s/{finished_token}")
        assert "Set-Cookie" not in page.headers


# ── logging ──────────────────────────────────────────────────────────────────

class TestLogsDoNotCarryCapabilities:
    def test_the_match_id_is_digested(self, app, spotify, db, caplog):
        import logging

        from conftest import (
            give_library, match_id_from, oauth_roundtrip, seed_features, track_ids,
        )

        seed_features(db, track_ids("x", 120))
        give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))

        with caplog.at_level(logging.INFO):
            created, _ = oauth_roundtrip(app.test_client(), "/login", "a-code")
        match_id = match_id_from(created)

        assert match_id not in caplog.text, "the join capability is in the log"
        assert security.digest(match_id) in caplog.text, "and now it is untraceable"

    def test_the_digest_is_short_and_not_reversible(self):
        d = security.digest("a" * 32)
        assert len(d) == 12
        assert d != "a" * 12
        assert security.digest("x") != security.digest("y")


# ── transport ────────────────────────────────────────────────────────────────

class TestHttpsIsForced:
    """Built on a bare Flask app, because the real one is configured local."""

    @pytest.fixture
    def deployed(self):
        from flask import Flask

        a = Flask(__name__)
        a.secret_key = "x"
        security.install(a, local=False)

        @a.route("/")
        def home():
            return "ok"

        return a.test_client()

    def test_plain_http_is_redirected_permanently(self, deployed):
        page = deployed.get("/", base_url="http://lovify.example")
        assert page.status_code == 301
        assert page.headers["Location"].startswith("https://")

    def test_https_is_served(self, deployed):
        page = deployed.get("/", base_url="https://lovify.example")
        assert page.status_code == 200
        assert page.headers["Strict-Transport-Security"] == security.HSTS

    def test_a_proxy_terminating_tls_is_believed(self, deployed):
        """Railway and Fly speak plain http to the app. Without this the
        app redirects to a URL the proxy sends straight back, forever."""
        page = deployed.get("/", base_url="http://lovify.example",
                            headers={"X-Forwarded-Proto": "https"})
        assert page.status_code == 200

    def test_local_development_is_not_redirected(self):
        from flask import Flask

        a = Flask(__name__)
        a.secret_key = "x"
        security.install(a, local=True)

        @a.route("/")
        def home():
            return "ok"

        page = a.test_client().get("/", base_url="http://127.0.0.1:8080")
        assert page.status_code == 200
        assert "Strict-Transport-Security" not in page.headers


# ── the policy against the actual templates ──────────────────────────────────

class TestThePolicyMatchesWhatThePagesLoad:
    """A content security policy is only as good as its agreement with the
    HTML. A blocked stylesheet or font produces no error anyone sees -- the
    page just renders slightly wrong, forever.

    This caught a real one: cdnjs serves Font Awesome's *stylesheet* to the
    older pages, and the first draft of the policy allowed that host for
    scripts only.
    """

    # Which directive governs each kind of reference, and the pattern that
    # identifies it in the markup.
    KINDS = (
        ("script-src", r'<script[^>]+src="(https://[^"]+)"'),
        ("style-src", r'<link[^>]+rel="stylesheet"[^>]+href="(https://[^"]+)"'),
        ("style-src", r'<link[^>]+href="(https://[^"]+)"[^>]+rel="stylesheet"'),
    )

    def test_every_external_resource_is_permitted(self):
        import re
        from pathlib import Path

        policy = directives(security.content_security_policy(local=True))
        templates = Path(__file__).resolve().parents[1] / "templates"

        unpermitted = []
        for page in sorted(templates.glob("*.html")):
            html = page.read_text()
            for directive, pattern in self.KINDS:
                for url in re.findall(pattern, html):
                    host = url.split("/")[2]
                    allowed = policy[directive].split()
                    if not any(host == a.removeprefix("https://").lstrip("*.")
                               or a.removeprefix("https://").startswith("*.")
                               and host.endswith(a.removeprefix("https://*."))
                               for a in allowed if a.startswith("https://")):
                        unpermitted.append(f"{page.name}: {directive} blocks {host}")

        assert not unpermitted, "\n".join(unpermitted)

    def test_no_template_carries_an_inline_handler_or_script_block(self):
        """script-src has no 'unsafe-inline', so either of these is dead code
        that looks alive. Put it in a file under static/ instead."""
        import re
        from pathlib import Path

        templates = Path(__file__).resolve().parents[1] / "templates"
        offenders = []
        for page in sorted(templates.glob("*.html")):
            html = page.read_text()
            # Strip Jinja comments first: they discuss this rule by name.
            html = re.sub(r"\{#.*?#\}", "", html, flags=re.S)
            if re.search(r"\son[a-z]+\s*=", html):
                offenders.append(f"{page.name}: inline event handler")
            if re.search(r"<script(?![^>]*\ssrc=)[^>]*>", html):
                offenders.append(f"{page.name}: inline <script> block")

        assert not offenders, "\n".join(offenders)
