"""Transport, headers, and CSRF.

Everything here is defence the application cannot provide by being correct.
The match flow can be flawless and still leak a result through a referrer
header, or be driven by a page on someone else's domain, or be served over
plain http on a coffee-shop network.

Three groups:

1. **Transport.** Force https and ask the browser to remember it (HSTS), so
   a session cookie carrying a participant's identity is never on the wire
   in the clear.
2. **Headers.** A content security policy that will not execute injected
   script, plus the smaller headers that close clickjacking, MIME sniffing,
   and referrer leakage.
3. **CSRF.** A signed token on the state-changing POSTs. `SameSite=Lax`
   already blocks the cross-site form post this guards against, in every
   browser that honours it; the token is the layer that does not depend on
   the browser getting that right.

Deliberately NOT here: anything that rate-limits or blocks by IP. This is a
25-user allowlisted Spotify app behind a dev-mode gate, and a blocklist we
cannot test is worse than none.
"""

from __future__ import annotations

import hmac
import secrets
from hashlib import sha256

from flask import redirect, request, session

CSRF_SESSION_KEY = "csrf"
CSRF_FIELD = "csrf_token"

# One year, which is what makes a domain eligible for preloading. Only ever
# sent over https, and only in deployment -- an HSTS header picked up from a
# localhost response pins 127.0.0.1 to https in that browser for a year and
# breaks every other local project the developer owns.
HSTS = "max-age=31536000; includeSubDomains"


def content_security_policy(local: bool) -> str:
    """The policy, assembled so each exception has to justify itself.

    `script-src` carries no 'unsafe-inline'. That is the directive that
    actually stops an injected `<script>`, so nothing in this project may
    use an inline handler or an inline script block; `static/lovify.js`
    exists because of this rule, not by preference.

    `style-src` does carry it, for one reason: the component bars on the
    result page set their width with a `style` attribute, and the width is
    a number computed per match. Splitting that into a stylesheet means
    generating CSS per request, which is a worse trade than permitting
    inline style on a page with no inline script.
    """
    policy = {
        "default-src": "'self'",
        "script-src": "'self' https://cdnjs.cloudflare.com https://sdk.scdn.co",
        # cdnjs is here as well as in script-src: it serves Font Awesome's
        # *stylesheet* to the older pages, which is easy to miss because the
        # same host appears above. A blocked stylesheet produces no error
        # anyone sees, just a page with the icons quietly missing.
        "style-src": ("'self' 'unsafe-inline' https://fonts.googleapis.com "
                      "https://cdnjs.cloudflare.com"),
        "font-src": "'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com",
        # Profile pictures. Spotify serves them from i.scdn.co, and from
        # Facebook's CDN for accounts created through Facebook login, which
        # is not a host we get to choose.
        "img-src": ("'self' data: https://i.scdn.co https://*.scdn.co "
                    "https://*.fbcdn.net https://platform-lookaside.fbsbx.com"),
        "connect-src": "'self'",
        # The OAuth handoff is a GET redirect, not a form post, but naming
        # Spotify here keeps the directive honest if that ever changes.
        "form-action": "'self' https://accounts.spotify.com",
        # Clickjacking. A public share card is exactly the kind of page
        # someone would frame invisibly over a "claim your prize" button.
        "frame-ancestors": "'none'",
        "base-uri": "'none'",
        "object-src": "'none'",
    }
    parts = [f"{k} {v}" for k, v in policy.items()]
    if not local:
        parts.append("upgrade-insecure-requests")
    return "; ".join(parts)


def install(app, local: bool) -> None:
    """Attach the transport guard and the response headers."""

    @app.before_request
    def _force_https():
        if local or request.is_secure:
            return None
        # Behind a proxy (Railway, Fly, Heroku) the connection to the app is
        # plain http even when the browser's connection is https. Trust the
        # header the proxy sets, or every request redirects forever.
        if request.headers.get("X-Forwarded-Proto", "http") == "https":
            return None
        return redirect(request.url.replace("http://", "https://", 1), code=301)

    @app.after_request
    def _headers(response):
        response.headers["Content-Security-Policy"] = content_security_policy(local)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"          # pre-CSP browsers
        # The public card links out to X, Facebook and WhatsApp. Without
        # this, the full share URL -- a bearer token -- travels to them in
        # the Referer header. Send the origin only.
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), camera=(), microphone=(), payment=(), usb=()")
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if not local:
            response.headers["Strict-Transport-Security"] = HSTS
        return response


# ── CSRF ─────────────────────────────────────────────────────────────────────

def token() -> str:
    """This browser's CSRF token, minted once per session."""
    existing = session.get(CSRF_SESSION_KEY)
    if not existing:
        existing = secrets.token_urlsafe(24)
        session[CSRF_SESSION_KEY] = existing
    return existing


def valid(submitted: str | None) -> bool:
    expected = session.get(CSRF_SESSION_KEY)
    if not expected or not submitted:
        return False
    # Constant time: a timing oracle on a CSRF token is a slow one, but the
    # comparison costs nothing to get right.
    return hmac.compare_digest(str(expected), str(submitted))


def digest(value: str) -> str:
    """A short, stable, non-reversible stand-in for a secret, for logs.

    Match ids and share tokens are bearer capabilities: anyone who reads one
    out of a log file can use it. They still need to be traceable through a
    log, so log this instead of the thing itself.
    """
    return sha256(value.encode()).hexdigest()[:12]
