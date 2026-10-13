"""Spotify OAuth and the one data call we make, behind an injectable seam.

Everything that touches the network lives here so the rest of the app can be
tested without a single HTTP request. That is not tidiness: the app's Spotify
credentials have already earned a 20-hour rate-limit ban once, and a test
suite that quietly calls the real API would do it again.

Scope note: `user-top-read`, `user-library-read` and `user-read-recently-played`,
all of which the registered app already holds, so no re-authorisation is
needed. All three are required, not optional. Top tracks alone return only
~110 unique ids across the three time ranges, and roughly 60% of those resolve
against the feature store -- about 66 usable tracks, below the 100 the scorer
needs before it will report a number. Saved songs are what carry a normal
listener over that line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# Spotify's own wording for the development-mode allowlist failure is opaque
# ("User not registered in the Developer Dashboard"). This is the single most
# likely failure on demo day, so it gets a message that says what to do.
NOT_ALLOWLISTED = (
    "Spotify refused the login. While the app is in development mode, every "
    "person who signs in has to be added to the allowlist by email in the "
    "Spotify developer dashboard, and it allows at most 5 people. Ask the "
    "project owner to add your Spotify account email, then try again."
)


class SpotifyNotConfigured(Exception):
    """No client id/secret. A deployment problem, not a user problem."""


class SpotifyError(Exception):
    """A Spotify call failed in a way the user should hear about."""

    def __init__(self, message: str, *, allowlist: bool = False):
        super().__init__(message)
        self.allowlist = allowlist


@dataclass
class SpotifyProfile:
    spotify_id: str
    display_name: str
    photo_url: str = ""


class SpotifyClient:
    """Real client. Injected, so tests substitute FakeSpotifyClient."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str,
                 scope: str = "user-top-read user-library-read user-read-recently-played",
                 session=None, timeout: float = 20.0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope
        self.timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    # ── step 1: send the user to Spotify ─────────────────────────────────────
    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def authorize_url(self, state: str) -> str:
        if not self.configured:
            # Without this the app happily builds `…/authorize?client_id=&…`
            # and sends the user to Spotify, who answer with a bare white
            # page reading "client_id: Not present". Nothing in that page
            # mentions this app, this machine, or .env, so the obvious
            # conclusion is that the app is broken.
            #
            # The usual cause is not a missing value in .env at all -- it is
            # a container started BEFORE .env was filled in, because compose
            # reads the file once at startup and never again.
            raise SpotifyNotConfigured(
                "This server has no Spotify credentials, so there is nothing "
                "to log in to yet.")
        return f"{AUTH_URL}?" + urlencode({
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "state": state,
            # Forces the account chooser. Without it the second person on a
            # shared laptop silently reuses the first person's Spotify session
            # and matches them against themselves.
            "show_dialog": "true",
        })

    # ── step 2: swap the code for a token we use once and discard ────────────
    def exchange_code(self, code: str) -> str:
        r = self.session.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout)

        if not getattr(r, "ok", False):
            body = (getattr(r, "text", "") or "")[:300]
            raise SpotifyError(f"Could not complete the Spotify login. {body}")
        token = r.json().get("access_token")
        if not token:
            raise SpotifyError("Spotify did not return an access token.")
        return token

    # ── step 3: the only data we actually want ───────────────────────────────
    def profile(self, token: str) -> SpotifyProfile:
        r = self.session.get(f"{API}/me", headers=self._auth(token),
                             timeout=self.timeout)
        if getattr(r, "status_code", 0) == 403:
            raise SpotifyError(NOT_ALLOWLISTED, allowlist=True)
        if not getattr(r, "ok", False):
            raise SpotifyError("Could not read your Spotify profile.")
        me = r.json()
        images = me.get("images") or []
        return SpotifyProfile(
            spotify_id=me.get("id", ""),
            display_name=me.get("display_name") or "Someone",
            photo_url=images[0]["url"] if images else "",
        )

    def top_track_ids(self, token: str, limit: int = 50,
                      time_range: str = "short_term") -> list[str]:
        """Top tracks for one time range. Caps at 50 per call, so we page."""
        return self._paged(f"{API}/me/top/tracks", token, limit,
                           extra={"time_range": time_range},
                           pick=lambda it: it.get("id"),
                           what="top tracks")

    def saved_track_ids(self, token: str, limit: int = 300) -> list[str]:
        """Saved ("liked") songs. This is where the volume comes from.

        Top tracks alone cannot fill a library: three time ranges overlap
        heavily and yield only ~110 unique ids, of which ~60% resolve against
        the feature store -- about 66 usable, under the 100 the scorer needs.
        Saved songs are paginated and unbounded, so they are what actually
        gets a normal listener over the line.
        """
        return self._paged(f"{API}/me/tracks", token, limit,
                           pick=lambda it: (it.get("track") or {}).get("id"),
                           what="saved songs")

    def recently_played_ids(self, token: str, limit: int = 50) -> list[str]:
        """Recent plays. A small top-up; the endpoint caps at 50 and has no offset."""
        return self._paged(f"{API}/me/player/recently-played", token,
                           min(limit, 50),
                           pick=lambda it: (it.get("track") or {}).get("id"),
                           what="recently played", paginate=False)

    def collect_library(self, token: str, budget: int = 400) -> dict:
        """Every track id we have scope to read, deduplicated.

        Returns {"track_ids": [...], "sources": {name: count}} so the caller
        can record where a person's library actually came from. We have never
        measured this for a real Spotify user, and it is the biggest unknown
        in the design -- so it gets recorded rather than discarded.

        Ordering is deliberate: top tracks first (most representative of
        current taste), then saved songs, then recent plays. Deduplication
        keeps the first occurrence, so the most meaningful signal survives
        truncation at the budget.

        A failure in any one source is tolerated. A user with no saved songs,
        or a brand-new account with no listening history, should still get as
        far as their data allows rather than seeing an error.
        """
        collected: list[str] = []
        sources: dict[str, int] = {}

        def take(name: str, fn):
            if len(collected) >= budget:
                sources[name] = 0
                return
            try:
                ids = fn(budget - len(collected))
            except SpotifyError as exc:
                if exc.allowlist:        # never swallow the allowlist failure
                    raise
                sources[name] = 0
                return
            before = len(collected)
            collected.extend(ids)
            # dedupe in place, preserving first-seen order
            collected[:] = list(dict.fromkeys(collected))
            sources[name] = len(collected) - before

        for tr in ("short_term", "medium_term", "long_term"):
            take(f"top_{tr}", lambda n, tr=tr: self.top_track_ids(token, min(n, 50), tr))
        take("saved", lambda n: self.saved_track_ids(token, n))
        take("recently_played", lambda n: self.recently_played_ids(token, n))

        return {"track_ids": collected[:budget], "sources": sources}

    # ── paging helper ────────────────────────────────────────────────────────
    def _paged(self, url: str, token: str, limit: int, pick, what: str,
               extra: dict | None = None, paginate: bool = True) -> list[str]:
        ids: list[str] = []
        offset = 0
        while len(ids) < limit:
            page = min(50, limit - len(ids))
            params = {"limit": page}
            if paginate:
                params["offset"] = offset
            params.update(extra or {})
            r = self.session.get(url, headers=self._auth(token), params=params,
                                 timeout=self.timeout)
            if getattr(r, "status_code", 0) == 403:
                raise SpotifyError(NOT_ALLOWLISTED, allowlist=True)
            if not getattr(r, "ok", False):
                raise SpotifyError(f"Could not read your {what} from Spotify.")
            items = r.json().get("items") or []
            if not items:
                break
            ids.extend(i for i in (pick(it) for it in items if it) if i)
            offset += len(items)
            if len(items) < page or not paginate:
                break
        return list(dict.fromkeys(ids))

    @staticmethod
    def _auth(token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}


@dataclass
class FakeSpotifyClient:
    """Test double. Deterministic, offline, and able to fail on demand."""

    profiles: dict = field(default_factory=dict)       # code -> SpotifyProfile
    tracks: dict = field(default_factory=dict)         # code -> top track ids
    saved: dict = field(default_factory=dict)          # code -> saved track ids
    recent: dict = field(default_factory=dict)         # code -> recent track ids
    fail_with: SpotifyError | None = None
    redirect_uri: str = "http://localhost/callback"
    calls: list = field(default_factory=list)

    # The real client exposes this so the app can refuse to send anyone to
    # Spotify with an empty client id. A double that does not carry the same
    # property is a double that lets that bug through the tests.
    configured: bool = True

    def authorize_url(self, state: str) -> str:
        if not self.configured:
            raise SpotifyNotConfigured("Fake client marked unconfigured.")
        return f"https://spotify.test/authorize?state={state}"

    def exchange_code(self, code: str) -> str:
        self.calls.append(("exchange", code))
        if self.fail_with:
            raise self.fail_with
        if code not in self.profiles:
            raise SpotifyError("unknown code")
        return f"token-for-{code}"

    def profile(self, token: str) -> SpotifyProfile:
        self.calls.append(("profile", token))
        if self.fail_with:
            raise self.fail_with
        return self.profiles[token.removeprefix("token-for-")]

    def top_track_ids(self, token: str, limit: int = 50,
                      time_range: str = "short_term") -> list[str]:
        self.calls.append(("top", token, time_range))
        if self.fail_with:
            raise self.fail_with
        return list(self.tracks.get(token.removeprefix("token-for-"), []))[:limit]

    def saved_track_ids(self, token: str, limit: int = 300) -> list[str]:
        self.calls.append(("saved", token))
        if self.fail_with:
            raise self.fail_with
        return list(self.saved.get(token.removeprefix("token-for-"), []))[:limit]

    def recently_played_ids(self, token: str, limit: int = 50) -> list[str]:
        self.calls.append(("recent", token))
        if self.fail_with:
            raise self.fail_with
        return list(self.recent.get(token.removeprefix("token-for-"), []))[:limit]

    def collect_library(self, token: str, budget: int = 400) -> dict:
        """Same contract as the real client, including per-source counts."""
        collected: list[str] = []
        sources: dict[str, int] = {}
        for name, fn in (("top_short_term", self.top_track_ids),
                         ("saved", self.saved_track_ids),
                         ("recently_played", self.recently_played_ids)):
            if self.fail_with:
                raise self.fail_with
            before = len(collected)
            collected.extend(fn(token, budget - len(collected)))
            collected[:] = list(dict.fromkeys(collected))
            sources[name] = len(collected) - before
        return {"track_ids": collected[:budget], "sources": sources}
