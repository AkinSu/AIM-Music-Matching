"""Which "Cancer" playlists are actually about Cancer.

MEASURED AGAINST THE LIVE API, which is why this module exists at all. Taking
the top results for each sign's name and trusting them produces two poisoned
classes:

    cancer -> "pov: my mom has cancer", "Cancer fighting songs",
              "Cancer Survivor Fight Songs", "Healing Cancer Frequencies",
              "POWERFUL KILLING CANCER CELLS WITH SOUND 10000HZ+741"
              -- 1 of 9 sampled results was astrological.

    leo    -> "LEO DAN MIX", "CARIN LEON MIX", "LEONARDO FAVIO EXITOS",
              "BEST OF LEON BRIDGES", "LOS INVASORES DE NUEVO LEON"
              -- 1 of 9. Mostly artists named Leo/Leon, heavily Latin.

Unfiltered, the classifier does not learn astrology. It learns "Cancer =
cancer-ward ballads" and "Leo = regional Mexican", and it learns them well,
because those are real and highly separable signals. A model can score far
above chance on that data and mean nothing by it.

A SECOND, LARGER MEASUREMENT (85-131 real titles per sign, four query
templates) settled how aggressive to be, and changed two things:

  * It CONFIRMED the lenient default. Requiring an emoji or an astrology word
    for every sign cuts keeps by ~40% per sign (Virgo 85->57, Pisces 83->48,
    Aquarius 78->48), and the titles it discards are overwhelmingly genuine:
    "Taurus Playlist (Made by a Taurus)", "she's a scorpio", "the real pisces
    playlist", "Can you tell I'm a Pisces?". Throwing those away to catch a
    handful of collisions is a bad trade, so only AMBIGUOUS_SIGNS have to
    prove themselves. `strict=True` remains, for measuring the alternative.

  * It RETRACTED "virgo is 9 of 9 clean". Nine titles was too small a sample.
    A larger one turns up "Best Of Romain Virgo" (a Jamaican reggae singer)
    and "VIRGO'S GROOVE" (a Beyonce track, so playlists named after it are
    about the song). Two more signs collide the same way: "Grupo Libra", a
    Mexican band, accounts for 5 of the 11 at-risk Libra titles, and "Gemini
    Home Entertainment" is a horror web series with a soundtrack following.

Those three are handled with TARGETED PATTERNS in SIGN_REJECT rather than by
adding the signs to AMBIGUOUS_SIGNS, because strict mode on Libra would also
kill "songs for a libra", which is genuine and sits in the same at-risk list.
A blocklist entry is precise where strict mode is a blunt instrument.

The healing-frequency genre is a separate hazard and applies to every sign:
"...10000HZ+741" playlists are sine tones, not music. Their audio features
are extreme and consistent, so they are exactly the rows a feature-based model
would latch onto.

EVERY RULE HERE IS AN EMPIRICAL GUESS and the tables are meant to be edited.
scripts/label_playlists.py exists to turn the guesses into measured precision
and recall against hand-labelled titles; until that is run, treat the numbers
this filter produces as provisional.

PRECEDENCE (first match wins, and the order is itself a decision):
    1. universal rejects (healing frequencies)   -- beat even a sign emoji,
       because "cancer energy 528hz" is still sine tones;
    2. per-sign rejects (disease context, name collisions);
    3. too many signs in one title -> a 12-sign mega-playlist is not evidence
       for any single sign;
    4. placement (Aries MOON, Taurus VENUS) -> kept but TAGGED, since a moon
       placement is a different concept from a sun sign and mixing them
       silently is the kind of thing that makes a dataset unexplainable.
       "aries sun placement" is NOT a placement -- see PLACEMENT_GENERIC;
    5. keep, on a sign emoji or on the sign word next to an astrology term;
    6. otherwise reject for want of astrological context.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from . import schema as S

# ── sign vocabulary ───────────────────────────────────────────────────────────
# The emoji is the single strongest signal we found: genuine zodiac playlists
# use it constantly and contaminated ones never do. Matched on the base
# codepoint, because titles carry it both bare (♋) and with the emoji
# variation selector (♋️ = U+264B U+FE0F).
ZODIAC_EMOJI = {
    "Aries": "♈", "Taurus": "♉", "Gemini": "♊", "Cancer": "♋",
    "Leo": "♌", "Virgo": "♍", "Libra": "♎", "Scorpio": "♏",
    "Sagittarius": "♐", "Capricorn": "♑", "Aquarius": "♒",
    "Pisces": "♓",
}

# Words that, next to a sign's name, mean the title is about astrology.
ASTRO_TERMS = (
    "season", "energy", "vibes", "vibe", "zodiac", "sign", "signs", "astrology",
    "astro", "horoscope", "placement", "placements", "moon", "sun", "rising",
    "venus", "mars", "mercury", "ascendant", "birth chart", "natal",
)

# Terms meaning the title is about a PLACEMENT rather than a sun sign. "sun"
# is deliberately absent: "aries sun" IS the sun sign.
PLACEMENT_BODIES = (
    "moon", "rising", "ascendant", "venus", "mars", "mercury", "jupiter",
    "saturn",
)
# "placement(s)" on its own is ambiguous -- "aries sun placement" is a sun
# sign described as a placement, while "aries placements" is not. So the
# generic word only tags a placement when "sun" is absent.
PLACEMENT_GENERIC = ("placement", "placements")

# Signs whose name is so heavily overloaded that the WHOLE class is poisoned
# without positive astrological evidence: "cancer" is a disease, "leo" is a
# given name. Measured contamination was ~8 of 9 for both.
#
# Deliberately NOT extended to Libra, Virgo or Gemini. Those collide with one
# or two specific entities, which SIGN_REJECT removes by name; demanding
# astrological context for a whole sign costs ~40% of its genuine playlists
# to remove a handful of bad ones. A sign belongs here only when its name is
# unusable on its own, not when it merely has a namesake.
AMBIGUOUS_SIGNS = frozenset({"Cancer", "Leo"})

# A title naming three signs is a 12-sign compilation, not evidence for one.
MAX_SIGNS_IN_TITLE = 2

# The 4,078-track "cancer energy♋️💥" playlist is real. One playlist that size
# supplies more rows than a hundred ordinary ones and would dominate its class
# on its own; under 15 tracks there is nothing to learn and Jaccard goes
# haywire (a 2-track playlist is a perfect duplicate of any other holding the
# same pair).
MIN_TRACKS = 15
MAX_TRACKS = 500


def _words(terms) -> re.Pattern:
    """Word-boundary alternation. Boundaries matter more than they look:
    'leon' contains 'leo', so substring matching would call LEON BRIDGES an
    astrology playlist."""
    body = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)")


# ── reject tables ─────────────────────────────────────────────────────────────
# (reason, pattern). Add rows here rather than adding code anywhere else.

UNIVERSAL_REJECT: tuple[tuple[str, re.Pattern], ...] = (
    # "...WITH SOUND 10000HZ+741": sine tones. No leading \b on hz, because in
    # "10000hz" there is no boundary between the digit and the h.
    ("frequency_healing", re.compile(
        r"(?:\d+\s*hz(?!\w)|(?<!\w)hz(?!\w)|frequenc(?:y|ies)|solfeggio|binaural"
        r"|isochronic|(?<!\w)(?:432|528|741|963)(?!\w))")),
)

SIGN_REJECT: dict[str, tuple[tuple[str, re.Pattern], ...]] = {
    "Cancer": (
        # NOTE: "healing" was deliberately removed from this table. It was the
        # one term here carrying real false-positive risk -- "cancer season
        # heals me" is a plausible genuine title -- and it was doing little
        # work, because the actual tone playlists ("Healing Cancer
        # Frequencies") are caught by frequency_healing first. A disease-shaped
        # title with no astrology word still fails Cancer's AMBIGUOUS_SIGNS
        # requirement, so this table only has to catch titles carrying BOTH.
        ("disease_context", _words((
            "survivor", "survivors", "fighting", "fight", "fighter", "chemo",
            "chemotherapy", "awareness", "diagnosis", "diagnosed", "remission",
            "tumor", "tumour", "oncology", "patient",
            "breast", "lung", "ribbon", "my mom", "my mum", "my dad",
            "my sister", "my brother", "rip", "memorial", "hospice",
        ))),
    ),
    "Leo": (
        # Artists and places. The word-boundary rule already stops LEON
        # BRIDGES and NUEVO LEON from matching the sign at all; these catch
        # the ones that really do say "leo".
        ("name_collision", re.compile(
            r"(?<!\w)leo\s+(?:dan|santana|rojas|mattioli|messi)(?!\w)"
            r"|carin\s+leon|leon\s+bridges|leonardo|leonid|nuevo\s+leon"
            r"|(?<!\w)leo\s+(?:garcia|moracchioli)(?!\w)")),
    ),
    # Grupo Libra, a Mexican band: 5 of the 11 at-risk Libra titles measured.
    # Separators are loose because the real titles use all of them -- "Grupo
    # Libra - Puras Romanticas", "El Grupo Libra-Exitos Inmortales",
    # "Ladron,grupo libra,Mandingo,..." -- and the frontman is credited in
    # front of the band name as often as not.
    "Libra": (
        ("name_collision", re.compile(r"grupo[\s\-_]*libra")),
    ),
    # Romain Virgo (reggae singer), and VIRGO'S GROOVE -- a Beyonce track, so
    # playlists named for it are about the song rather than the sign.
    "Virgo": (
        ("name_collision", re.compile(
            r"romain[\s\-_]*virgo|virgo'?s?[\s\-_]+groove")),
    ),
    # Gemini Home Entertainment, a horror web series with a soundtrack
    # following. "gemini radio" is deliberately NOT blocked: a sign radio
    # playlist is an ordinary genuine title and blocking it would cost more
    # than it saves.
    "Gemini": (
        ("name_collision", re.compile(r"gemini[\s\-_]*home[\s\-_]*entertainment")),
    ),
}

ASTRO_RE = _words(ASTRO_TERMS)
PLACEMENT_RE = _words(PLACEMENT_BODIES)
PLACEMENT_GENERIC_RE = _words(PLACEMENT_GENERIC)
SUN_RE = _words(("sun",))
SIGN_WORD_RE = {s: _words((s.lower(),)) for s in S.ZODIAC_SIGNS}


def rebuild_patterns() -> None:
    """Recompile the matchers from the tables above.

    The patterns are compiled once at import, so editing a table at RUNTIME
    (a notebook trying a new term, a test) would otherwise have no effect and
    look like the rule was wrong rather than unapplied. Editing the source
    needs nothing; this is for the interactive case.
    """
    global ASTRO_RE, PLACEMENT_RE, PLACEMENT_GENERIC_RE, SUN_RE, SIGN_WORD_RE
    ASTRO_RE = _words(ASTRO_TERMS)
    PLACEMENT_RE = _words(PLACEMENT_BODIES)
    PLACEMENT_GENERIC_RE = _words(PLACEMENT_GENERIC)
    SUN_RE = _words(("sun",))
    SIGN_WORD_RE = {s: _words((s.lower(),)) for s in S.ZODIAC_SIGNS}

KEEP, REJECT, PLACEMENT = "keep", "reject", "placement"


# ── normalisation ─────────────────────────────────────────────────────────────

_QUOTES = {ord(c): "'" for c in "‘’ʼ´`′"}


def normalise_title(name: str) -> str:
    """Casefold, strip accents, drop emoji variation selectors.

    Accent stripping is not cosmetic: "LEONARDO FAVIO ÉXITOS" and
    "CARIN LEÓN" are real titles, and a rule table written in ASCII has to
    match them. Curly apostrophes fold for the same reason -- "VIRGO'S GROOVE"
    arrives with either quote character depending on the curator's keyboard,
    and a table cannot carry both spellings of every rule.
    """
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(c for c in text
                   if not unicodedata.combining(c) and c != "️")
    return " ".join(text.translate(_QUOTES).casefold().split())


def _signs_mentioned(text: str, raw: str) -> set[str]:
    return {s for s in S.ZODIAC_SIGNS
            if SIGN_WORD_RE[s].search(text) or ZODIAC_EMOJI[s] in raw}


# ── the classifier ────────────────────────────────────────────────────────────

def classify_playlist_title(name: str, sign: str | None,
                            strict: bool | None = None) -> tuple[str, str]:
    """-> ("keep" | "reject" | "placement", reason).

    `sign=None` means "no sign expected" -- the control set. Only the
    universal rules apply there, because a control playlist is supposed to be
    called "shower songs".

    `strict` controls whether the sign's name alone is enough. It defaults to
    `sign in AMBIGUOUS_SIGNS`: Cancer and Leo must show an emoji or an
    astrology word, the other ten are believed on their name. Pass strict=True
    to demand astrological context everywhere -- stricter, and it costs real
    data for the ten signs that measured clean, so the CLI reports how many
    playlists that choice would drop.
    """
    raw = str(name or "")
    text = normalise_title(raw)

    for reason, pattern in UNIVERSAL_REJECT:
        if pattern.search(text):
            return REJECT, reason

    if sign is None:
        return KEEP, "control"

    for reason, pattern in SIGN_REJECT.get(sign, ()):
        if pattern.search(text):
            return REJECT, reason

    has_emoji = ZODIAC_EMOJI[sign] in raw
    has_word = bool(SIGN_WORD_RE[sign].search(text))
    if not (has_emoji or has_word):
        return REJECT, "sign_not_in_title"

    if len(_signs_mentioned(text, raw)) > MAX_SIGNS_IN_TITLE:
        return REJECT, "multi_sign"

    placement = PLACEMENT_RE.search(text) or (
        None if SUN_RE.search(text) else PLACEMENT_GENERIC_RE.search(text))
    if placement:
        return PLACEMENT, f"placement:{placement.group(0)}"

    if has_emoji:
        return KEEP, "sign_emoji"
    if ASTRO_RE.search(text):
        return KEEP, "astro_context"

    strict = (sign in AMBIGUOUS_SIGNS) if strict is None else strict
    if strict:
        return REJECT, "no_astro_context"
    return KEEP, "sign_in_title"


def classify_playlist(name: str, sign: str | None, n_tracks: int | None = None,
                      strict: bool | None = None,
                      min_tracks: int = MIN_TRACKS,
                      max_tracks: int = MAX_TRACKS) -> tuple[str, str]:
    """Title rules first, then size.

    Title first on purpose: when the 4,078-track "cancer energy♋️💥" playlist
    is dropped we would rather the audit log said why the TITLE was or was not
    believable, and the size rule is unambiguous enough that it needs no
    diagnosis.
    """
    cls, reason = classify_playlist_title(name, sign, strict=strict)
    if cls == REJECT:
        return cls, reason
    if n_tracks is not None:
        if n_tracks < min_tracks:
            return REJECT, "too_few_tracks"
        if n_tracks > max_tracks:
            return REJECT, "too_many_tracks"
    return cls, reason


# ── frame-level application ───────────────────────────────────────────────────

def classify_frame(df: pd.DataFrame, strict: bool | None = None,
                   control: bool = False, **size_kw) -> pd.DataFrame:
    """One row per playlist: id, sign, name, n_tracks, title_class, reason.

    Takes either shape: a row-per-track scrape frame, or a frame that is
    already one row per playlist with an `n_tracks` column (an audit file).
    Both re-run the CURRENT rule tables, so an audit sheet drawn after a table
    edit reflects the edit rather than the verdict stored on disk.

    On a scrape frame, size is the DISTINCT TRACKS WE ACTUALLY HOLD rather
    than the playlist object's `tracks.total`, so the decision matches the
    data on disk even when the scrape was capped by max_tracks_per_playlist.
    """
    if df.empty:
        return pd.DataFrame(columns=[S.PLAYLIST_ID, S.STAR_SIGN, S.PLAYLIST_NAME,
                                     "n_tracks", S.TITLE_CLASS, S.TITLE_REASON])
    if S.TRACK_ID in df.columns:
        per = (df.groupby(S.PLAYLIST_ID, sort=True)
                 .agg(**{S.STAR_SIGN: (S.STAR_SIGN, "first"),
                         S.PLAYLIST_NAME: (S.PLAYLIST_NAME, "first"),
                         "n_tracks": (S.TRACK_ID, "nunique")})
                 .reset_index())
    elif "n_tracks" in df.columns:
        per = (df[[S.PLAYLIST_ID, S.STAR_SIGN, S.PLAYLIST_NAME, "n_tracks"]]
               .drop_duplicates(S.PLAYLIST_ID).reset_index(drop=True))
    else:
        raise ValueError(
            f"need either {S.TRACK_ID} (a scrape frame) or n_tracks "
            f"(an audit frame); got {list(df.columns)[:8]}")
    verdicts = [
        classify_playlist(r[S.PLAYLIST_NAME], None if control else r[S.STAR_SIGN],
                          int(r["n_tracks"]), strict=strict, **size_kw)
        for _, r in per.iterrows()
    ]
    per[S.TITLE_CLASS] = [v[0] for v in verdicts]
    per[S.TITLE_REASON] = [v[1] for v in verdicts]
    return per


def apply_filter(df: pd.DataFrame, strict: bool | None = None,
                 control: bool = False, drop_placements: bool = False,
                 **size_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (kept rows tagged with title_class/title_reason, per-playlist verdicts).

    Placements are KEPT and tagged by default. They are genuine astrology, just
    a different question from a sun sign, and a column is recoverable where a
    silent drop is not.
    """
    verdicts = classify_frame(df, strict=strict, control=control, **size_kw)
    if verdicts.empty:
        return df.assign(**{S.TITLE_CLASS: "keep", S.TITLE_REASON: "empty"}), verdicts
    keep = {KEEP} if drop_placements else {KEEP, PLACEMENT}
    tags = verdicts.set_index(S.PLAYLIST_ID)[[S.TITLE_CLASS, S.TITLE_REASON]]
    out = df.copy()
    out[S.TITLE_CLASS] = out[S.PLAYLIST_ID].map(tags[S.TITLE_CLASS])
    out[S.TITLE_REASON] = out[S.PLAYLIST_ID].map(tags[S.TITLE_REASON])
    return out[out[S.TITLE_CLASS].isin(keep)].reset_index(drop=True), verdicts


def rejection_breakdown(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Playlists dropped per reason, with an example title each.

    The example is the point: a reason with a wrong-looking title next to it is
    a rule to go and edit, which is not visible from a count alone.
    """
    if verdicts.empty:
        return pd.DataFrame(columns=[S.TITLE_CLASS, S.TITLE_REASON, "playlists",
                                     "tracks", "example"])
    grouped = (verdicts.groupby([S.TITLE_CLASS, S.TITLE_REASON])
               .agg(playlists=(S.PLAYLIST_ID, "nunique"),
                    tracks=("n_tracks", "sum"),
                    example=(S.PLAYLIST_NAME, "first"))
               .reset_index())
    return grouped.sort_values(["playlists"], ascending=False).reset_index(drop=True)


def strictness_cost(df: pd.DataFrame) -> dict:
    """What demanding astrological context for ALL signs would cost.

    Reported rather than assumed, because it is the one filter decision where
    the measurements point two ways: Cancer and Leo need the stricter rule and
    the other ten signs do not.
    """
    lenient = classify_frame(df, strict=False)
    strict = classify_frame(df, strict=True)
    kept_l = set(lenient.loc[lenient[S.TITLE_CLASS] != REJECT, S.PLAYLIST_ID])
    kept_s = set(strict.loc[strict[S.TITLE_CLASS] != REJECT, S.PLAYLIST_ID])
    only_lenient = lenient[lenient[S.PLAYLIST_ID].isin(kept_l - kept_s)]
    return {
        "kept_lenient": len(kept_l),
        "kept_strict": len(kept_s),
        "would_drop": len(kept_l - kept_s),
        "would_drop_tracks": int(only_lenient["n_tracks"].sum()),
        "by_sign": only_lenient.groupby(S.STAR_SIGN)[S.PLAYLIST_ID].nunique().to_dict(),
    }
