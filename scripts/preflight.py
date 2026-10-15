#!/usr/bin/env python3
"""Check everything the REAL app needs, before you waste time on a login loop.

Run `make check`. Every line is either OK or tells you the exact fix.

This exists because the failure modes of a real Spotify run are all silent
or misleading. A redirect URI that differs by one character gives you
"INVALID_CLIENT: Invalid redirect URI" with no hint which one it saw. An
account that is not on the development allowlist gets a 403 that looks like
a bug in the app. An empty feature store scores everyone at zero.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "LOVIFY ME APP"

OK, WARN, BAD = "\033[32m  ok \033[0m", "\033[33mwarn \033[0m", "\033[31mFIX  \033[0m"
problems: list[str] = []


def say(level, label, detail=""):
    print(f"{level} {label}" + (f"\n       {detail}" if detail else ""))
    if level is BAD:
        problems.append(label)


def env() -> dict[str, str]:
    """Read .env without importing anything that might not be installed."""
    path = ROOT / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main() -> int:
    print("\n  Lovify preflight — the real app, real Spotify\n")
    e = env()

    # ── config ──────────────────────────────────────────────────────────────
    if not (ROOT / ".env").exists():
        say(BAD, "No .env file", "cp .env.example .env")
        return report()

    if e.get("SECRET_KEY"):
        say(OK, "SECRET_KEY is set")
    else:
        say(BAD, "SECRET_KEY is empty",
            'python3 -c "import secrets; print(secrets.token_hex(32))"  → put it in .env')

    base = e.get("BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    redirect = f"{base}/callback"

    if "localhost" in base:
        say(BAD, f"BASE_URL uses localhost ({base})",
            "Spotify REJECTS localhost. Use http://127.0.0.1:8080 — they are not "
            "interchangeable to Spotify even though they are to your browser.")
    else:
        say(OK, f"BASE_URL = {base}")

    cid, secret = e.get("SPOTIFY_CLIENT_ID"), e.get("SPOTIFY_CLIENT_SECRET")
    if cid and secret:
        say(OK, f"Spotify credentials present (client id ends …{cid[-4:]})")
    else:
        missing = " and ".join(n for n, v in
                               (("SPOTIFY_CLIENT_ID", cid),
                                ("SPOTIFY_CLIENT_SECRET", secret)) if not v)
        say(BAD, f"{missing} empty in .env",
            "https://developer.spotify.com/dashboard → your app → Settings")

    # All three, because collect_library() calls an endpoint for each. A
    # missing user-library-read gives a 403 on /v1/me/tracks that reads
    # exactly like the allowlist refusal, which cost an afternoon.
    scope = e.get("SCOPE", "")
    needed = {"user-top-read", "user-library-read", "user-read-recently-played"}
    missing = needed - set(scope.split())
    if not missing:
        say(OK, f"SCOPE = {scope}")
    else:
        say(BAD, f"SCOPE is missing {', '.join(sorted(missing))}",
            "SCOPE=user-top-read user-library-read user-read-recently-played\n       "
            "Then restart, and approve the Spotify consent screen again -- it\n       "
            "will now ask for your saved songs, which it did not before.")

    # ── the one that bites everyone ─────────────────────────────────────────
    print()
    say(WARN, "Both testers must be on the development allowlist",
        "Dashboard → your app → Settings → User Management → add each person's\n       "
        "Spotify ACCOUNT EMAIL. \033[1mFIVE maximum\033[0m -- not 25; Spotify cut it, and\n       "
        "both people in a match count. An account that is not on the list gets a\n       "
        "403 at login that looks exactly like a bug in this app.")

    # ── does the credential actually work ───────────────────────────────────
    print()
    if cid and secret:
        try:
            import base64
            import json
            import urllib.error
            import urllib.request

            auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
            req = urllib.request.Request(
                "https://accounts.spotify.com/api/token",
                data=b"grant_type=client_credentials",
                headers={"Authorization": f"Basic {auth}",
                         "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=15) as r:
                json.loads(r.read())["access_token"]
            say(OK, "Spotify accepted the client id and secret")
        except urllib.error.HTTPError as exc:
            say(BAD, f"Spotify REJECTED the credentials ({exc.code})",
                "The client id or secret in .env is wrong, or was rotated. "
                "Copy both again from the dashboard.")
        except Exception as exc:                      # offline, DNS, proxy
            say(WARN, f"Could not reach Spotify to check the credentials ({exc})")

    # ── the redirect URI, checked against Spotify rather than assumed ───────
    if cid:
        print()
        check_redirect(cid, secret, redirect)

    # ── model and data ──────────────────────────────────────────────────────
    print()
    artifact = ROOT / "ml" / "artifacts" / "lovify_models.joblib"
    if artifact.exists():
        say(OK, f"Model artifact present ({artifact.stat().st_size / 1e6:.1f} MB)")
    else:
        say(BAD, "No model artifact at ml/artifacts/lovify_models.joblib",
            "Without it the app runs but every match says 'scoring is not "
            "available on this server yet'.")

    url = os.getenv("DATABASE_URL") or e.get("DATABASE_URL")
    if url:
        try:
            sys.path.insert(0, str(APP))
            from sqlalchemy import func, select

            from models import TrackFeature, make_engine
            engine = make_engine(url)
            with engine.connect() as conn:
                n = conn.execute(
                    select(func.count()).select_from(TrackFeature.__table__)).scalar_one()
            if n > 1_000_000:
                say(OK, f"Feature store has {n:,} tracks")
            elif n:
                say(WARN, f"Feature store has only {n:,} tracks", "make seed")
            else:
                say(BAD, "Feature store is empty", "make seed")
        except Exception as exc:
            say(WARN, f"Could not check the database ({type(exc).__name__})",
                "Normal from the host — the database only listens inside Docker.\n"
                "       Check it with: make up && make check-db")
    else:
        say(WARN, "No DATABASE_URL here; compose supplies it inside the container")

    return report()


def check_redirect(client_id: str, secret: str, redirect: str) -> None:
    """Print the redirect URI to verify by eye. It cannot be checked for you.

    Two automated checks were tried here and both returned a confident
    false pass, which is worse than no check at all:

    1. GET /authorize and look for the error. When the caller is not logged
       in, Spotify answers with its login page and validates the redirect
       URI only afterwards. A logged-in browser got "redirect_uri: Not
       matching configuration" from the very URL this had just called ok.

    2. POST /api/token with a bogus code and read the error. Spotify checks
       the code FIRST, so a registered URI, an unregistered one, and
       http://totally-bogus.example/nope all return exactly
       {"error":"invalid_grant","error_description":"Invalid authorization
       code"}. Verified against all three.

    So this prints the string and says to compare it. Do not replace it with
    something that merely looks more thorough.
    """
    say(WARN, "Check this redirect URI is in the dashboard (cannot be verified "
              "from here -- see the note in scripts/preflight.py):",
        f"\033[1m{redirect}\033[0m\n       "
        "Dashboard -> your app -> Settings -> Edit -> Redirect URIs.\n       "
        "It must match character for character, then press \033[1mSave\033[0m.\n       "
        "Old entries can stay; extras do no harm.\n       "
        "Getting it wrong gives 'redirect_uri: Not matching configuration',\n       "
        "which never says which part it disliked.")


def report() -> int:
    print()
    if problems:
        print(f"  \033[31m{len(problems)} thing(s) to fix:\033[0m")
        for p in problems:
            print(f"    · {p}")
        print()
        return 1
    print("  \033[32mReady.\033[0m  make up   then open " +
          (env().get("BASE_URL") or "http://127.0.0.1:8080") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
