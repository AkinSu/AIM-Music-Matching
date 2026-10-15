"""Bridge between the web app and the `lovify_ml` library.

Three jobs, and the third is the interesting one.

1. Load the trained model **once**, at startup. `LovifyModels.load()` reads a
   joblib artifact and rebuilds a recommender; doing that per request would
   add seconds to every page.

2. Turn a list of Spotify track ids into the DataFrame the scorer expects, by
   looking the ids up in the `track_features` table. Only ~41% of real tracks
   resolve -- the audio features come from a pre-2021 dump because Spotify
   removed the live endpoint -- so a partial or empty result is normal, not
   an error.

3. **Degrade honestly when the artifact is missing.** `ml/artifacts/` is
   gitignored, so a fresh clone has the library but no trained model until
   someone builds one. That should not take the site down: the match flow,
   the OAuth round trip and the database are all independently useful. When
   scoring is unavailable the app says so in those words, rather than
   crashing or -- far worse -- inventing a number.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_SRC = REPO_ROOT / "ml" / "src"
DEFAULT_ARTIFACT = REPO_ROOT / "ml" / "artifacts" / "lovify_models.joblib"

# Below this many resolvable tracks the scorer refuses to produce a
# percentile. Mirrored from LovifyModels.MIN_LIBRARY_TRACKS so the UI can warn
# before a round trip; the library remains the authority.
MIN_LIBRARY_TRACKS = 100


@dataclass
class ScoringUnavailable:
    """Why scoring cannot run. Shown to the operator, not the end user."""

    reason: str

    def __bool__(self) -> bool:
        return False


class Scorer:
    """Wraps LovifyModels, or explains why it could not be loaded."""

    def __init__(self, artifact_path: str | os.PathLike | None = None,
                 models=None):
        self.artifact_path = Path(artifact_path or DEFAULT_ARTIFACT)
        self.models = models
        self.unavailable: ScoringUnavailable | None = None
        if models is None:
            self._load()

    def _load(self) -> None:
        if str(ML_SRC) not in sys.path:
            sys.path.insert(0, str(ML_SRC))
        try:
            from lovify_ml.service import LovifyModels  # noqa: WPS433
        except Exception as exc:
            self.unavailable = ScoringUnavailable(
                f"lovify_ml is not importable from {ML_SRC} ({exc.__class__.__name__}: "
                f"{exc}). The match flow still works; compatibility scoring does not."
            )
            return

        if not self.artifact_path.exists():
            self.unavailable = ScoringUnavailable(
                f"No model artifact at {self.artifact_path}. Build one with "
                "notebook 04 (end-to-end) and save it there."
            )
            return

        try:
            self.models = LovifyModels.load(self.artifact_path)
        except Exception as exc:
            self.unavailable = ScoringUnavailable(
                f"Could not load {self.artifact_path}: {exc.__class__.__name__}: {exc}"
            )

    @property
    def ready(self) -> bool:
        return self.models is not None

    @property
    def min_tracks(self) -> int:
        return getattr(self.models, "min_library_tracks", MIN_LIBRARY_TRACKS)

    # ── scoring ──────────────────────────────────────────────────────────────
    def compatibility(self, tracks_a, tracks_b) -> dict:
        """Score two track frames.

        Passes `status` straight through from the library. The caller must
        treat "insufficient_library" and "no_resolvable_tracks" as outcomes
        with their own screens -- they are the library declining to guess,
        which is the behaviour we want, not a failure to handle.
        """
        if not self.ready:
            return {
                "status": "scoring_unavailable",
                "percentile": None,
                "interpretation": "Compatibility scoring is not configured on this server.",
                "detail": self.unavailable.reason if self.unavailable else "unknown",
            }
        return self.models.compatibility(tracks_a, tracks_b)

    def taste_profile(self, tracks) -> dict | None:
        """One person's zodiac taste summary, or None if unavailable.

        The whole zodiac result of this project lives behind this call and
        was never wired to a page. It is a *summary*, not a prediction: the
        measured ability to predict a real birth sign from audio is 1.39x
        chance, below the threshold this project pre-registered as dead.
        What survives that is "your listening leans Scorpio-ish compared with
        an average listener", with the lift stated, which is what the model
        actually supports.

        Returns None rather than raising, because a missing taste summary
        must never take down a results page that has a real compatibility
        score on it.
        """
        if not self.ready or tracks is None or not len(tracks):
            return None
        try:
            return self.models.taste_profile(tracks)
        except Exception:                       # pragma: no cover - defensive
            import logging
            logging.getLogger(__name__).warning(
                "taste_profile failed; the page renders without it", exc_info=True)
            return None


def features_for(db_session, track_ids: list[str]):
    """Look up audio features for these track ids, as a DataFrame.

    Returns an empty frame when nothing resolves, which is a real and common
    outcome: coverage against the feature store runs about 41%, and the misses
    are not random -- they skew recent and obscure, because the dump predates
    2021 and nothing can be added to it now.
    """
    import pandas as pd

    from models import TrackFeature

    if not track_ids:
        return pd.DataFrame()

    rows = (db_session.query(TrackFeature)
            .filter(TrackFeature.track_id.in_(track_ids))
            .all())
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([r.as_row() for r in rows])
    # Preserve the user's own ordering (most-listened first); the scorer does
    # not care, but truncation elsewhere would otherwise silently reorder.
    rank = {t: i for i, t in enumerate(track_ids)}
    return (frame.assign(_rank=frame["track_id"].map(rank))
                 .sort_values("_rank")
                 .drop(columns="_rank")
                 .reset_index(drop=True))


def coverage(track_ids: list[str], resolved) -> float:
    """Share of a person's tracks we could find features for."""
    if not track_ids:
        return 0.0
    return len(resolved) / len(track_ids)
