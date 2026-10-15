"""The app <-> ML seam, against the real artifact and the real feature store.

Every other test in this directory uses StubScorer, because the joblib
artifact is gitignored and CI has no 1.4-million-row database. That is the
right default, and it left a hole: until this file, nothing had ever checked
that the trained model and the web app agree about anything. The first time
`LovifyModels` scored something shaped like a person was in a throwaway
container at the end of a review pass, and it found a bug on the first try --
`explain()` naming three shared genres for two people whose genre overlap was
exactly zero.

Skips unless BOTH of these are true, so a laptop with no data and CI are
unaffected:

  * ml/artifacts/lovify_models.joblib exists
  * LOVIFY_SEEDED_DATABASE_URL points at a database with a populated
    track_features table

Run it with:

    LOVIFY_SEEDED_DATABASE_URL=postgresql+psycopg://lovify:lovify@localhost:5433/lovify \
        python -m pytest tests/test_real_model.py -v

The seeded database is only ever READ here -- a separate engine, never
`lovify.engine`, so this cannot collide with the drop_all the other fixtures
do.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import sessionmaker

from models import TrackFeature
from scoring import DEFAULT_ARTIFACT, Scorer, coverage, features_for

SEEDED_URL = os.getenv("LOVIFY_SEEDED_DATABASE_URL")

# Two tastes that should not resemble each other. Genre strings come from the
# feature store's own vocabulary; if a rename ever empties one of these the
# fixture fails loudly rather than silently comparing two empty libraries.
METAL_JAZZ = ["metal", "alternative metal", "hard rock", "album rock", "jazz", "bebop"]
POP_LATIN = ["dance pop", "pop", "post-teen pop", "latin", "reggaeton", "urbano latino"]

LIBRARY = 260   # a plausible personal library after the 41% feature-store join


def _skip_reason() -> str | None:
    if not SEEDED_URL:
        return "set LOVIFY_SEEDED_DATABASE_URL to run against the real feature store"
    if not DEFAULT_ARTIFACT.exists():
        return f"no trained model at {DEFAULT_ARTIFACT}"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@pytest.fixture(scope="module")
def seeded():
    engine = create_engine(SEEDED_URL)
    if not inspect(engine).has_table("track_features"):
        pytest.skip("the seeded database has no track_features table")
    session = sessionmaker(bind=engine)()
    n = session.execute(select(func.count()).select_from(TrackFeature.__table__)).scalar_one()
    if n < 100_000:
        pytest.skip(f"track_features holds only {n:,} rows; run `make seed` first")
    yield session
    session.close()


@pytest.fixture(scope="module")
def scorer():
    s = Scorer()
    assert s.ready, f"artifact exists but would not load: {s.unavailable and s.unavailable.reason}"
    return s


def _library(session, genres, limit=LIBRARY, offset=0):
    """A listener's worth of track ids, most popular first."""
    rows = session.execute(
        select(TrackFeature.track_id)
        .where(TrackFeature.genre.in_(genres), TrackFeature.danceability.isnot(None))
        .order_by(TrackFeature.popularity.desc().nullslast())
        .limit(limit).offset(offset)).scalars().all()
    assert len(rows) == limit, f"only {len(rows)} tracks for {genres[:2]}…; genre vocabulary changed?"
    return list(rows)


@pytest.fixture(scope="module")
def people(seeded):
    return {
        "metal_jazz": _library(seeded, METAL_JAZZ),
        "pop_latin": _library(seeded, POP_LATIN),
        # Same taste as metal_jazz, no track in common: the case that separates
        # "these two match" from "these two pasted the same library".
        "metal_jazz_twin": _library(seeded, METAL_JAZZ, offset=LIBRARY),
    }


@pytest.fixture(scope="module")
def frames(seeded, people):
    return {k: features_for(seeded, ids) for k, ids in people.items()}


def test_the_model_loads_in_the_app_the_app_actually_runs(scorer):
    assert scorer.min_tracks == 100


def test_ids_from_the_feature_store_all_resolve(people, frames):
    for name, frame in frames.items():
        assert coverage(people[name], frame) == 1.0, (
            f"{name}: ids came out of track_features and did not resolve back into it")


def test_opposite_tastes_score_low_and_alike_tastes_score_high(scorer, frames):
    """The only property a compatibility score must have to be worth shipping."""
    opposite = scorer.compatibility(frames["metal_jazz"], frames["pop_latin"])
    alike = scorer.compatibility(frames["metal_jazz"], frames["metal_jazz_twin"])
    for r in (opposite, alike):
        assert r["status"] == "ok", r

    assert opposite["percentile"] < alike["percentile"], (
        f"metal/jazz vs pop/latin scored {opposite['percentile']}, "
        f"metal/jazz vs itself-but-different-tracks scored {alike['percentile']}")
    assert alike["n_shared_artists"] > 0
    assert opposite["n_shared_artists"] == 0


def test_the_explanation_never_claims_an_overlap_the_score_denies(scorer, frames):
    """The bug this file was written after.

    `top_shared_genres` is rendered under the heading "Where you overlap".
    When genre_overlap is 0.0 there is nothing to put there, and three
    arbitrary labels is not an approximation of nothing -- it is a false
    statement about two strangers.
    """
    r = scorer.compatibility(frames["metal_jazz"], frames["pop_latin"])
    assert r["components"]["genre_overlap"] == 0.0
    assert r["top_shared_genres"] == []
    assert r["shared_artists"] == []


def test_a_light_listener_is_refused_rather_than_guessed_at(scorer, frames, seeded, people):
    """99 resolvable tracks is below the threshold; the scorer must say so."""
    thin = features_for(seeded, people["metal_jazz"][:99])
    r = scorer.compatibility(thin, frames["pop_latin"])
    assert r["status"] == "insufficient_library"
    assert r["percentile"] is None
    assert r["interpretation"]


def test_nothing_resolvable_is_its_own_answer(scorer, frames):
    import pandas as pd
    r = scorer.compatibility(pd.DataFrame(), frames["pop_latin"])
    assert r["status"] == "no_resolvable_tracks"
    assert r["percentile"] is None


def test_the_score_grades_rather_than_flipping_between_0_and_100(scorer, seeded, people):
    """A score that only ever returns 0 or 100 is a coin, not a measurement.

    Blends one library into the other in steps and checks the percentile
    moves down without going back up. Measured run: 100 -> 100 -> 100 -> 99
    -> 35 -> 5.5 -> 0.
    """
    base = features_for(seeded, people["metal_jazz"])
    metal, pop = people["metal_jazz_twin"], people["pop_latin"]

    seen = []
    for swapped in (0, 50, 100, 150, 200):
        mixed = metal[: LIBRARY - swapped] + pop[:swapped]
        r = scorer.compatibility(base, features_for(seeded, mixed))
        assert r["status"] == "ok", r
        seen.append(r["percentile"])

    assert seen == sorted(seen, reverse=True), f"not monotone as tastes diverge: {seen}"
    assert len(set(seen)) >= 3, f"only {len(set(seen))} distinct values across the sweep: {seen}"
    assert seen[0] > seen[-1]
