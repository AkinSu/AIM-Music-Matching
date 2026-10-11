"""The survey: does a person's real birth sign predict their real music taste?

This is the only part of the project with ground truth. The zodiac playlist
model learns from crowd labels that random strangers disagree about; the
archetype model is a mapping we designed ourselves and so cannot be wrong in
any testable way. Here the label is a fact about a person, and the question
has an answer.

We expect that answer to be no. That is the point: a clean, adequately
powered null is a real finding, and it is only credible because the analysis
was fixed in advance. See PREREGISTRATION.md.

Pipeline:

    load_responses(csv)        Google Forms export -> tidy frame
    sign_from_birthday(m, d)   month/day -> sign, computed not self-reported
    parse_artists(text)        messy pasted text -> list of artist names
    resolve_artists(names)     names -> genres, from cache/snapshot/API
    build_taste_vectors(...)   -> one genre distribution per respondent

Everything is offline by default. The Spotify app has already earned one
20-hour ban, so network access is opt-in and cache-first.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as Fm
from . import schema as S

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
ARTIST_SNAPSHOT = RAW / "artist_genres.csv"
RESOLVE_CACHE = RAW / "survey_artist_cache.json"


# ── birth date -> sign ───────────────────────────────────────────────────────
#
# Tropical sun-sign dates. Sources disagree by a day at most cusps because the
# true solar ingress drifts a few hours a year and the published tables round
# differently. There is no "correct" set; what matters is that we fix ONE
# convention, state it, and apply it to everybody.
#
# Convention used here (the most commonly published set, e.g. Britannica's):
# each entry is the first day of that sign.
#
# Consequence worth knowing: people born within a day of a cusp may disagree
# with us about their own sign. That is not an error to suppress -- the
# agreement rate between the computed sign and the self-reported one is one of
# the things the analysis reports, and cusp disagreement is most of it.

_SIGN_STARTS: list[tuple[int, int, str]] = [
    (1, 20, "Aquarius"),
    (2, 19, "Pisces"),
    (3, 21, "Aries"),
    (4, 20, "Taurus"),
    (5, 21, "Gemini"),
    (6, 21, "Cancer"),
    (7, 23, "Leo"),
    (8, 23, "Virgo"),
    (9, 23, "Libra"),
    (10, 23, "Scorpio"),
    (11, 22, "Sagittarius"),
    (12, 22, "Capricorn"),
]


def sign_from_birthday(month: int, day: int) -> str:
    """Sun sign for a birth month and day, by the convention documented above.

    Dates before 20 January fall to Capricorn, which is the only sign that
    straddles the year boundary.

    >>> sign_from_birthday(3, 20), sign_from_birthday(3, 21)
    ('Pisces', 'Aries')
    >>> sign_from_birthday(1, 1), sign_from_birthday(12, 31)
    ('Capricorn', 'Capricorn')
    """
    month, day = int(month), int(day)
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range: {month}")
    if not 1 <= day <= 31:
        raise ValueError(f"day out of range: {day}")

    current = "Capricorn"  # anything before 20 January
    for m, d, sign in _SIGN_STARTS:
        if (month, day) >= (m, d):
            current = sign
        else:
            break
    return current


# ── free-text artist parsing ─────────────────────────────────────────────────
#
# Respondents paste whatever Spotify's share button gave them, or retype it
# from memory. Observed shapes include numbered lists, comma runs, one per
# line, "1. Artist — 1,234 plays", bullet characters, emoji, and stray
# "and"s. Parsing this leniently matters more than parsing it cleverly: a
# dropped artist is a silently thinner taste vector, which biases the result
# toward the null we are already expecting to find.

_LEAD_NUMBER = re.compile(r"^\s*(?:\d{1,3}\s*[.)\-:]|[-*•·–—])\s*")
_TRAILING_STATS = re.compile(
    r"\s*[—–\-:|(]\s*(?:\d[\d,.]*\s*)?"
    r"(?:plays?|streams?|minutes?|mins?|hours?|hrs?|songs?|listens?)\b.*$",
    re.IGNORECASE,
)
_SPLIT_LINES = re.compile(r"[\n\r;]+")


def _strip_emoji(text: str) -> str:
    """Drop pictographs but keep letters, marks, punctuation and spacing.

    Category-based rather than a range list, so it survives scripts we did
    not anticipate -- respondents will paste Korean, Japanese and Arabic
    artist names and those must come through intact.
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"So", "Sk", "Cf", "Cs", "Co"}
    )


def parse_artists(text: str, max_artists: int = 60) -> list[str]:
    """Pull artist names out of a pasted blob.

    Lenient by design. Splits on newlines and semicolons first; only falls
    back to splitting a line on commas when that line has no other structure,
    because plenty of real artist names contain commas ("Tyler, The Creator"
    being the obvious one this project keeps tripping over).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    out: list[str] = []
    for raw_line in _SPLIT_LINES.split(_strip_emoji(text)):
        line = _LEAD_NUMBER.sub("", raw_line).strip()
        line = _TRAILING_STATS.sub("", line).strip()
        if not line:
            continue

        # A single line holding several artists: split on commas only when it
        # looks like a list rather than one comma-containing name. Two or more
        # commas, or a comma plus " and ", reads as a list.
        parts = [line]
        if line.count(",") >= 2 or (line.count(",") == 1 and re.search(r"\band\b", line, re.I)):
            parts = re.split(r",| and ", line, flags=re.IGNORECASE)

        for part in parts:
            name = part.strip().strip("\"'`.,")
            name = re.sub(r"\s{2,}", " ", name)
            if len(name) < 2 or len(name) > 80:
                continue
            if name.lower() in {"and", "the", "etc", "n/a", "na", "none", "idk"}:
                continue
            if name not in out:
                out.append(name)

    return out[:max_artists]


def normalise_artist(name: str) -> str:
    """Key for matching a pasted name against the snapshot."""
    name = unicodedata.normalize("NFKD", _strip_emoji(str(name)))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"^\s*the\s+", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


# ── resolving artists to genres ──────────────────────────────────────────────

class ArtistResolver:
    """Names -> genre, cache first, snapshot second, network last.

    The order is deliberate. `/v1/search` costs a request per unresolved name
    and this app has already been banned for 20 hours once; the 2021 snapshot
    covers 305,595 artists for free and, for a survey about mainstream
    listening, will resolve most of what people paste.
    """

    def __init__(self, client=None, cache_path: Path | None = RESOLVE_CACHE,
                 snapshot_path: Path | None = ARTIST_SNAPSHOT,
                 allow_network: bool = False):
        self.client = client
        self.allow_network = allow_network and client is not None
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

        self.snapshot: dict[str, dict] = {}
        if snapshot_path and Path(snapshot_path).exists():
            snap = pd.read_csv(snapshot_path)
            for name, genre in zip(snap["name"], snap["genre"]):
                key = normalise_artist(name)
                if key and key not in self.snapshot:
                    self.snapshot[key] = {"name": str(name), "genre": str(genre),
                                          "source": "snapshot"}

        self.stats = {"cache": 0, "snapshot": 0, "network": 0, "unresolved": 0}

    def resolve_one(self, name: str) -> dict | None:
        key = normalise_artist(name)
        if not key:
            return None
        if key in self.cache:
            self.stats["cache"] += 1
            return self.cache[key]
        if key in self.snapshot:
            self.stats["snapshot"] += 1
            self.cache[key] = self.snapshot[key]
            return self.cache[key]
        if self.allow_network:
            found = self._search(name)
            if found:
                self.stats["network"] += 1
                self.cache[key] = found
                return found
        self.stats["unresolved"] += 1
        return None

    def _search(self, name: str) -> dict | None:
        page = self.client.get("/search", {"q": name, "type": "artist", "limit": 1})
        items = (page or {}).get("artists", {}).get("items") or []
        if not items:
            return None
        a = items[0]
        genres = a.get("genres") or []
        return {"name": a.get("name", name), "genre": genres[0] if genres else "unknown",
                "artist_id": a.get("id"), "popularity": a.get("popularity"),
                "source": "network"}

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1))


def resolve_artists(names, client=None, cache=None, allow_network: bool = False):
    """Convenience wrapper: list of names -> list of resolution dicts."""
    resolver = ArtistResolver(client=client, cache_path=cache,
                              allow_network=allow_network)
    return [r for r in (resolver.resolve_one(n) for n in names) if r], resolver


# ── Google Forms ingest ──────────────────────────────────────────────────────
#
# Forms exports every column header as the full question text, so matching on
# a keyword is the only thing robust to Akin rewording a question after the
# first responses land.

_COLUMN_PATTERNS: list[tuple[str, str]] = [
    ("timestamp", r"timestamp"),
    ("birth_month", r"birth.*month|month.*born"),
    ("birth_day", r"birth.*day|day.*(of|you were) born|day of the month"),
    ("self_sign", r"what.*(star|zodiac|sun).*sign|your sign"),
    ("artists_text", r"top artists|favourite artists|favorite artists|artists"),
    ("songs_text", r"top songs|favourite songs|favorite songs|songs"),
    ("belief", r"believe|belief|astrology.*scale|how much.*astrology"),
    ("library_size", r"how many songs|library size|size of your library"),
    ("source", r"where did you|how did you (find|hear)|referr?al"),
]

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _month_to_int(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text.isdigit():
        n = int(text)
        return n if 1 <= n <= 12 else np.nan
    for name, number in _MONTHS.items():
        if text.startswith(name[:3]):
            return number
    return np.nan


def _first_int(value) -> float:
    if pd.isna(value):
        return np.nan
    m = re.search(r"\d{1,2}", str(value))
    return int(m.group()) if m else np.nan


def load_responses(csv_path, verbose: bool = True) -> pd.DataFrame:
    """Read a Google Forms CSV export into a tidy, validated frame.

    Returns one row per usable respondent with: respondent_id, computed
    `star_sign`, `self_sign`, `artists` (parsed list), `belief`, plus whatever
    optional fields were answered. Drops are counted and explained rather than
    silently applied -- a survey that quietly discards a third of its
    responses is not a survey.
    """
    raw = pd.read_csv(csv_path)
    mapped: dict[str, str] = {}
    for field, pattern in _COLUMN_PATTERNS:
        for col in raw.columns:
            if col in mapped.values():
                continue
            if re.search(pattern, str(col), re.IGNORECASE):
                mapped[field] = col
                break

    missing = [f for f in ("birth_month", "birth_day", "artists_text")
               if f not in mapped]
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in {list(raw.columns)}. "
            "Either the form wording changed a lot, or this is the wrong export.")

    df = pd.DataFrame({"respondent_id": [f"R{i:04d}" for i in range(len(raw))]})
    df["birth_month"] = raw[mapped["birth_month"]].map(_month_to_int)
    df["birth_day"] = raw[mapped["birth_day"]].map(_first_int)
    df["artists"] = raw[mapped["artists_text"]].map(parse_artists)
    for field in ("self_sign", "belief", "library_size", "source", "songs_text"):
        if field in mapped:
            df[field] = raw[mapped[field]].values
    if "belief" in df.columns:
        df["belief"] = df["belief"].map(_first_int)
    if "self_sign" in df.columns:
        df["self_sign"] = df["self_sign"].astype(str).str.strip().str.title()

    n0 = len(df)
    bad_date = df["birth_month"].isna() | df["birth_day"].isna()
    no_artists = df["artists"].map(len) == 0
    df = df[~(bad_date | no_artists)].copy()

    df[S.STAR_SIGN] = [sign_from_birthday(m, d)
                       for m, d in zip(df["birth_month"], df["birth_day"])]
    df["n_artists"] = df["artists"].map(len)

    if verbose:
        print(f"[survey] {n0} responses -> {len(df)} usable")
        if bad_date.sum():
            print(f"[survey]   dropped {bad_date.sum()} with an unreadable birth date")
        if no_artists.sum():
            print(f"[survey]   dropped {no_artists.sum()} with no parsable artists")
        print(f"[survey]   median artists per respondent: {df['n_artists'].median():.0f}")
    return df.reset_index(drop=True)


# ── taste vectors ────────────────────────────────────────────────────────────

def to_long(responses: pd.DataFrame, resolver: ArtistResolver) -> pd.DataFrame:
    """Explode respondents into one row per (respondent, artist) with a genre.

    Shaped like `listens` elsewhere in the project so `features` and
    `similarity` work on it unchanged -- `user_id`, `artist`, `genre`.
    """
    rows = []
    for _, r in responses.iterrows():
        for name in r["artists"]:
            hit = resolver.resolve_one(name)
            rows.append({
                S.USER_ID: r["respondent_id"],
                S.ARTIST: (hit or {}).get("name", name),
                S.GENRE: (hit or {}).get("genre", "unknown"),
                S.POPULARITY: (hit or {}).get("popularity", np.nan),
                "resolved": hit is not None,
            })
    return pd.DataFrame(rows)


def build_taste_vectors(responses: pd.DataFrame, resolver: ArtistResolver | None = None):
    """One genre distribution per respondent. Returns (vectors, long_frame)."""
    resolver = resolver or ArtistResolver()
    long = to_long(responses, resolver)
    known = long[long[S.GENRE] != "unknown"]
    vectors = Fm.genre_profile(known if len(known) else long)
    return vectors, long


# ── what the survey can and cannot score ─────────────────────────────────────
#
# CompatibilityScorer's default blend wants audio features, release years and
# popularity per track. Respondents give us ARTISTS, so three of the five
# components are simply not computable here:
#
#   genre_overlap        yes  -- from resolved artist genres
#   artist_jaccard       yes  -- the artists themselves
#   obscurity_match      yes  -- artist popularity, where resolved
#   feature_wasserstein  NO   -- needs per-track audio features
#   era_overlap          NO   -- needs per-track release years
#
# Renormalised over the three that survive. This is a real limitation of
# asking for artists rather than a library export, and it means the survey's
# pairwise test is a weaker instrument than the one used on scraped data.
SURVEY_WEIGHTS = {
    "genre_overlap": 0.50,
    "artist_jaccard": 0.35,
    "obscurity_match": 0.15,
    "feature_wasserstein": 0.0,
    "era_overlap": 0.0,
}


class SurveyScorer:
    """Pairwise similarity using only what artist-level data supports.

    `similarity.CompatibilityScorer` cannot be used directly here: its
    constructor scales S.AUDIO_FEATURES, which survey responses do not have,
    so it raises before scoring anything. Rather than feed it fake zero
    columns -- which would silently contribute a constant to every pair and
    look like a working metric -- this reuses the three component functions
    that are genuinely computable and renormalises over them.

    The component functions themselves are imported from `similarity`, not
    reimplemented, so the survey and the scraped-data analysis are measuring
    overlap the same way.
    """

    def __init__(self, long: pd.DataFrame, weights: dict[str, float] | None = None):
        from . import similarity as SIM

        self._sim = SIM
        self.weights = {k: v for k, v in (weights or SURVEY_WEIGHTS).items() if v > 0}
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        self.by_user = {u: g for u, g in long.groupby(S.USER_ID)}
        self.users = list(self.by_user)
        self.genre_profiles = Fm.genre_profile(long)
        self.artist_sets = {u: set(g[S.ARTIST]) for u, g in self.by_user.items()}
        self._has_pop = long[S.POPULARITY].notna().any()

    def components(self, u: str, v: str) -> dict[str, float]:
        out = {
            "genre_overlap": self._sim.genre_overlap(self.genre_profiles, u, v),
            "artist_jaccard": self._sim.artist_jaccard(self.artist_sets, u, v),
        }
        if "obscurity_match" in self.weights:
            out["obscurity_match"] = (
                self._sim.obscurity_match(self.by_user, u, v) if self._has_pop else 0.0)
        return out

    def raw_score(self, u: str, v: str) -> float:
        c = self.components(u, v)
        return float(sum(w * c.get(k, 0.0) for k, w in self.weights.items()))
