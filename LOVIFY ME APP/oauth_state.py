"""The OAuth `state` parameter, which now has two jobs instead of one.

Previously `state` carried a random uuid, stored in the session and compared
on return. That is textbook CSRF protection and it was correct.

The cross-device flow breaks it, because `state` now also has to carry *which
match is being joined*. The invitee's browser has never seen this server
before -- they arrived from a link in a text message -- so the match id cannot
come from their cookie. It has to survive the round trip through Spotify.

Putting the match id in `state` naively would create a real vulnerability: an
attacker who gets a victim to click a crafted authorize URL could redirect
the victim's Spotify identity into a match the *attacker* controls, and then
read the victim's listening data off the results page. The match id must
therefore be unforgeable, not merely present.

The scheme
----------
    state = nonce . match_id . role . signature

    signature = HMAC-SHA256(secret_key, "nonce.match_id.role"), truncated

Three independent checks on the way back, in this order:

1. **Signature.** Verified first, in constant time. A tampered match id or
   role fails here before anything else is trusted. This is what stops the
   attack above: only this server can mint a valid state, so a victim cannot
   be steered into a match of someone else's choosing.

2. **Nonce against the session.** The nonce was written to *this browser's*
   session immediately before the redirect. A state minted for one browser is
   useless in another, which is the original CSRF property, preserved intact.

3. **Single use.** The nonce is removed from the session once consumed, so a
   captured callback URL cannot be replayed.

What this deliberately does not prevent: anyone holding the share link can
join the match. That is what a share link is for. The property being defended
is narrower and more important -- that nobody can cause *someone else's*
login to land in a match they did not choose.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

SEPARATOR = "."
SIG_BYTES = 16          # 128 bits, ample for an un-replayable short-lived token
NONCE_BYTES = 16
SESSION_NONCE_KEY = "oauth_nonce"


class StateError(Exception):
    """Raised when a returned state cannot be trusted."""


def new_nonce() -> str:
    return secrets.token_hex(NONCE_BYTES)


def _signature(secret_key: str | bytes, payload: str) -> str:
    key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()[:SIG_BYTES * 2]


def build(secret_key: str | bytes, nonce: str, match_id: str, role: str) -> str:
    """Mint a signed state. Store `nonce` in the session before redirecting."""
    for part, name in ((nonce, "nonce"), (match_id, "match_id"), (role, "role")):
        if SEPARATOR in part:
            raise ValueError(f"{name} must not contain {SEPARATOR!r}: {part!r}")
    payload = SEPARATOR.join((nonce, match_id, role))
    return f"{payload}{SEPARATOR}{_signature(secret_key, payload)}"


def parse(secret_key: str | bytes, state: str | None,
          session_nonce: str | None) -> tuple[str, str]:
    """Validate a returned state and return (match_id, role).

    Raises StateError on anything suspicious. The caller should treat that as
    "start over", never as "continue with defaults".
    """
    if not state:
        raise StateError("no state returned from Spotify")

    parts = state.split(SEPARATOR)
    if len(parts) != 4:
        raise StateError("malformed state")
    nonce, match_id, role, sig = parts

    # 1. Signature first: nothing else in the state is trustworthy until the
    #    server has confirmed it minted this exact payload.
    expected = _signature(secret_key, SEPARATOR.join((nonce, match_id, role)))
    if not hmac.compare_digest(sig, expected):
        raise StateError("state signature did not verify")

    # 2. Bind to this browser. Without this, a valid state captured from
    #    anywhere could be replayed by anyone.
    if not session_nonce:
        raise StateError("no nonce in session; the login was not started here")
    if not hmac.compare_digest(nonce, session_nonce):
        raise StateError("state does not belong to this browser session")

    return match_id, role
