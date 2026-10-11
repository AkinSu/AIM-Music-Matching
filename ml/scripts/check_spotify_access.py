#!/usr/bin/env python
"""Probe exactly what your Spotify app can still do.

On 2024-11-27 Spotify removed several Web API endpoints for apps that did not
already hold extended access. The whole 2023 Lovify pipeline was built on one
of them (/v1/audio-features), so before planning anything we need to know,
empirically, which endpoints this app's credentials still open.

USAGE
    1. Put your credentials in  ml/.env   (that file is gitignored):

           SPOTIFY_CLIENT_ID=xxxxxxxxxxxxxxxxxxxx
           SPOTIFY_CLIENT_SECRET=xxxxxxxxxxxxxxxx

       Get them from https://developer.spotify.com/dashboard -> Lovify
       -> Settings -> "View client secret".

    2. Run:

           cd ml && .venv/bin/python scripts/check_spotify_access.py

Nothing is written anywhere and no user login is involved: this uses the
Client Credentials flow, which is an app-level token. That is also the flow
the zodiac scraper will use, so a green result here means the scrape is
viable.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

OK, DEAD, WARN = "\033[92m", "\033[91m", "\033[93m"
DIM, END = "\033[2m", "\033[0m"


def get_token(client_id: str, client_secret: str) -> str:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        ACCOUNTS,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {basic}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if not r.ok:
        print(f"{DEAD}Could not get an app token: {r.status_code} {r.text[:300]}{END}")
        print("\nUsual causes: wrong client id/secret, or the app was deleted.")
        sys.exit(1)
    return r.json()["access_token"]


def probe(session: requests.Session, label: str, path: str, params: dict | None = None,
          critical: bool = False) -> tuple[str, int, dict | None]:
    """Call one endpoint and classify the outcome."""
    try:
        r = session.get(f"{API}{path}", params=params or {}, timeout=30)
    except Exception as exc:  # network, DNS, timeout
        print(f"  {WARN}ERROR{END}  {label:<34} {exc}")
        return "error", 0, None

    if r.ok:
        mark, verdict = f"{OK}WORKS{END}", "works"
    elif r.status_code in (401, 403):
        mark, verdict = f"{DEAD} GONE{END}", "gone"
    elif r.status_code == 404:
        mark, verdict = f"{WARN}  404{END}", "not found"
    else:
        mark, verdict = f"{WARN}  {r.status_code}{END}", f"http {r.status_code}"

    flag = "  <-- CRITICAL" if critical and verdict != "works" else ""
    print(f"  {mark}  {label:<34} {DIM}{r.status_code}{END}{flag}")
    return verdict, r.status_code, (r.json() if r.ok else None)


PLACEHOLDERS = {"", "paste_your_client_id_here", "paste_your_secret_here",
                "your_id_here", "your_secret_here", "xxxxxxxxxxxxxxxxxxxx"}


def load_credentials() -> tuple[str | None, str | None]:
    """Prefer ml/.env, fall back to the Flask app's own .env.

    Placeholder values count as absent. Without that, a half-filled ml/.env
    silently shadows real credentials sitting in the app's .env -- plain
    load_dotenv() will not overwrite a variable that is already set, so the
    fallback never fires and you get a confusing auth failure instead.
    """
    def clean(v: str | None) -> str | None:
        v = (v or "").strip().strip('"').strip("'")
        return None if v in PLACEHOLDERS or v.startswith("paste_") else v or None

    for path in (ROOT / ".env", ROOT.parent / "LOVIFY ME APP" / ".env"):
        if not path.exists():
            continue
        vals = dotenv_values(path)
        cid, secret = clean(vals.get("SPOTIFY_CLIENT_ID")), clean(vals.get("SPOTIFY_CLIENT_SECRET"))
        if cid and secret:
            if path != ROOT / ".env":
                print(f"{DIM}using credentials from {path}{END}")
            return cid, secret

    # last resort: already-exported environment variables
    return clean(os.getenv("SPOTIFY_CLIENT_ID")), clean(os.getenv("SPOTIFY_CLIENT_SECRET"))


def main() -> None:
    cid, secret = load_credentials()
    if not cid or not secret:
        print(f"{DEAD}No credentials found.{END}\n")
        print(f"Create {ROOT / '.env'} containing:\n")
        print("    SPOTIFY_CLIENT_ID=your_id_here")
        print("    SPOTIFY_CLIENT_SECRET=your_secret_here\n")
        print("From https://developer.spotify.com/dashboard -> Lovify -> Settings.")
        sys.exit(1)

    print(f"\nclient id {cid[:6]}...{cid[-4:]}   (secret loaded, {len(secret)} chars)")
    token = get_token(cid, secret)
    print(f"{OK}app token acquired{END}  (Client Credentials flow)\n")

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})

    # Find live ids to probe with, rather than hardcoding ones that may rot.
    print("STILL EXPECTED TO WORK")
    _, _, tr = probe(s, "search (tracks)", "/search",
                     {"q": "love", "type": "track", "limit": 1}, critical=True)
    track_id = artist_id = None
    if tr and tr.get("tracks", {}).get("items"):
        t = tr["tracks"]["items"][0]
        track_id, artist_id = t["id"], t["artists"][0]["id"]
        print(f"         {DIM}probing with: {t['name']} - {t['artists'][0]['name']}{END}")

    _, _, pl = probe(s, "search (playlists)", "/search",
                     {"q": "aries", "type": "playlist", "limit": 5}, critical=True)
    playlist_id = None
    if pl and pl.get("playlists", {}).get("items"):
        items = [i for i in pl["playlists"]["items"] if i]
        if items:
            playlist_id = items[0]["id"]
            print(f"         {DIM}found playlist: {items[0]['name']!r} "
                  f"by {items[0]['owner']['display_name']}{END}")

    if playlist_id:
        probe(s, "playlist tracks", f"/playlists/{playlist_id}/tracks",
              {"limit": 5}, critical=True)
    if artist_id:
        probe(s, "artists (genres)", "/artists", {"ids": artist_id}, critical=True)
    if track_id:
        probe(s, "tracks (popularity, year)", "/tracks", {"ids": track_id}, critical=True)

    print("\nDEPRECATED 2024-11-27  (expected to fail on a non-extended app)")
    af_verdict = "skipped"
    if track_id:
        af_verdict, _, af = probe(s, "audio-features", "/audio-features", {"ids": track_id})
        probe(s, "audio-analysis", f"/audio-analysis/{track_id}")
        probe(s, "recommendations", "/recommendations", {"seed_tracks": track_id, "limit": 5})
    if artist_id:
        probe(s, "related-artists", f"/artists/{artist_id}/related-artists")
    probe(s, "featured-playlists", "/browse/featured-playlists", {"limit": 5})

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    if af_verdict == "works":
        print(f"{OK}audio-features STILL WORKS on this app.{END}")
        print("""
This app holds pre-deprecation access -- genuinely valuable, and rare.

  * Guard it. Do not delete the app, do not let it go inactive, and do not
    request a quota-mode change: a review could reset its status.
  * The scraper can pull features live for every track it finds, so the
    Kaggle store becomes a backup rather than the primary source.
  * Still build the feature-store path anyway. If this access ever lapses
    the whole pipeline dies with it, and the fallback is cheap insurance.""")
    elif af_verdict == "gone":
        print(f"{DEAD}audio-features is GONE on this app (as expected).{END}")
        print("""
This is the normal post-2024 state and the plan already assumes it.

  * Audio features come from a local store: the Kaggle 600k dump, joined on
    track_id. Put it at ml/data/raw/tracks.csv.
  * Expect partial coverage, and expect the misses to be non-random -- they
    cluster on obscure, non-Western and very recent tracks, which is exactly
    the music that makes a listener distinctive. Log coverage per user and
    treat it as a bias measurement, not a nuisance.
  * The good news from notebook 02: genre overlap and artist Jaccard beat
    every audio-feature metric, and both still work. The signal was never
    really in the features Spotify took away.""")
    else:
        print(f"{WARN}audio-features: {af_verdict}{END} -- inspect the output above.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
