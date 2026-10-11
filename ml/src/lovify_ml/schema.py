"""The data contract.

Every notebook and module agrees on these column names, so a DataFrame built
from synthetic data and one built from the Kaggle dump are interchangeable.
"""

from __future__ import annotations

# ── Audio features ────────────────────────────────────────────────────────────
# These are the 13 fields Spotify's /v1/audio-features endpoint returned.
# NOTE: that endpoint was deprecated 2024-11-27. See spotify.py for what this
# means in practice. The Kaggle 600k dataset is now our source for these.

BOUNDED_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]  # all already on [0, 1]

UNBOUNDED_FEATURES = [
    "tempo",        # ~50-200 BPM
    "loudness",     # ~-60-0 dB
    "duration_ms",  # ~60k-600k
]  # these must be scaled or they dominate every distance

CATEGORICAL_FEATURES = [
    "key",             # 0-11, pitch class -- cyclic, not ordinal
    "mode",            # 0/1 minor/major
    "time_signature",  # 3,4,5...
]

# The default modelling feature set: continuous only. `key` is deliberately
# excluded -- it is cyclic (key 11 is adjacent to key 0), so feeding it as an
# integer teaches the model a false ordering. See features.encode_key().
AUDIO_FEATURES = BOUNDED_FEATURES + UNBOUNDED_FEATURES

ALL_FEATURES = AUDIO_FEATURES + CATEGORICAL_FEATURES

# ── Identity / metadata columns ───────────────────────────────────────────────
TRACK_ID = "track_id"
TRACK_NAME = "track_name"
ARTIST = "artist"
ALBUM = "album"
GENRE = "genre"
POPULARITY = "popularity"
YEAR = "year"

META_COLUMNS = [TRACK_ID, TRACK_NAME, ARTIST, ALBUM, GENRE, POPULARITY, YEAR]

# ── Zodiac ────────────────────────────────────────────────────────────────────
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
N_SIGNS = len(ZODIAC_SIGNS)
CHANCE_ACCURACY = 1.0 / N_SIGNS  # 0.0833 -- the number every result is judged against

STAR_SIGN = "star_sign"
PLAYLIST_NAME = "playlist_name"

# ── Users ─────────────────────────────────────────────────────────────────────
USER_ID = "user_id"


def feature_bounds() -> dict[str, tuple[float, float]]:
    """Plausible (min, max) per feature -- used to sanity-check real data."""
    bounds = {f: (0.0, 1.0) for f in BOUNDED_FEATURES}
    bounds.update(
        tempo=(0.0, 250.0),
        loudness=(-60.0, 5.0),
        duration_ms=(5_000.0, 1_800_000.0),
        key=(0, 11),
        mode=(0, 1),
        time_signature=(0, 7),
    )
    return bounds


# ── Playlist provenance (written by scrape.py) ────────────────────────────────
# The 2023 scrape kept only (track, sign). That made every later question
# unanswerable: who made this playlist, how many people follow it, and is it
# an independent opinion or a copy of the playlist above it in the results?
# These columns exist so those questions can be asked of the data itself.

PLAYLIST_ID = "playlist_id"
PLAYLIST_OWNER_ID = "playlist_owner_id"
PLAYLIST_OWNER = "playlist_owner"
PLAYLIST_FOLLOWERS = "playlist_followers"
PLAYLIST_DESCRIPTION = "playlist_description"
PLAYLIST_SNAPSHOT = "playlist_snapshot_id"
PLAYLIST_TRACK_TOTAL = "playlist_track_total"
TRACK_POSITION = "track_position"   # index of the track within its playlist
ADDED_AT = "added_at"
SEARCH_QUERY = "search_query"       # the query string that surfaced the playlist
SEARCH_RANK = "search_rank"         # its 0-based position in that query's results
SCRAPED_AT = "scraped_at"
ARTIST_ID = "artist_id"             # primary artist, for a later genre lookup

PROVENANCE_COLUMNS = [
    PLAYLIST_ID, PLAYLIST_NAME, PLAYLIST_OWNER_ID, PLAYLIST_OWNER,
    PLAYLIST_FOLLOWERS, PLAYLIST_DESCRIPTION, PLAYLIST_SNAPSHOT,
    PLAYLIST_TRACK_TOTAL, SEARCH_QUERY, SEARCH_RANK, SCRAPED_AT,
]

# ── Playlist independence (written by dedup.py) ───────────────────────────────
# One cluster of near-duplicate playlists is ONE opinion, however many copies
# of it the search returned. See dedup.py's module docstring.

PLAYLIST_CLUSTER = "playlist_cluster"
LABEL_CONFIDENCE = "label_confidence"
N_CLUSTERS_SIGN = "n_clusters_sign"    # clusters of THIS sign holding the track
N_CLUSTERS_TOTAL = "n_clusters_total"  # clusters of ANY sign holding the track

# The primary artist's FULL genre list, pipe-separated. GENRE keeps just the
# first one (what the models use); this keeps the rest because the rest is
# free -- it arrives in the same /artists response -- and discarding four of
# an artist's five genres cannot be undone later.
#
# Not kept because genre was shown to be the strongest similarity signal:
# that notebook-02 finding has been withdrawn as circular (the synthetic
# users were built from genre mixtures and truth was the distance between
# those mixtures). Genre is a feature that still survives the 2024-11-27
# deprecation, which is a different and smaller claim.
ARTIST_GENRES = "artist_genres"

# ── Title filtering (written by filters.py) ───────────────────────────────────
# "Cancer" and "Leo" searches return cancer-ward playlists and artists called
# Leon. title_class records what the filter decided and title_reason records
# why, so the decision is auditable instead of baked into the row set.

TITLE_CLASS = "title_class"      # keep | reject | placement
TITLE_REASON = "title_reason"

# ── Negative control (written by scrape.scrape_control) ───────────────────────
# Non-astrological playlists carrying uniformly random fake sign labels. If
# the real zodiac data scores no better than this does, there is no signal.

IS_CONTROL = "is_control"
