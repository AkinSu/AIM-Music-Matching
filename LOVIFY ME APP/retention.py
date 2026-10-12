"""Forget things on a schedule.

ARCHITECTURE.md has promised a 7-day sweep since the data model was written
and nothing implemented it, so every abandoned half-match has been holding
one person's listening history indefinitely. The cheapest security property
available to this project is not having the data.

Three rules, in order of how much they throw away:

1. An **expired** match (nobody joined within MATCH_TTL) loses its track
   lists. Nothing can be scored from it any more, so the lists are pure
   liability.
2. A **scored** match loses its track lists too, after a shorter window. The
   result is already computed and stored; the raw library was only ever an
   input. Both people keep their percentile and their shared artists.
3. A match older than the retention horizon is **deleted outright**, result
   and share cards included.

Run it from `make sweep`, or on a daily cron in deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from models import Match, MatchStatus, Participant, ShareCard, utcnow

# A scored match's raw track lists are dead weight the moment the result is
# stored. A day's grace in case scoring has to be re-run during an incident.
TRACKS_TTL = timedelta(days=1)

# How long a match exists at all. Long enough that someone can come back and
# find it in their history; short enough that this is not an archive.
MATCH_RETENTION = timedelta(days=180)


@dataclass
class SweepReport:
    expired: int = 0
    tracks_cleared: int = 0
    deleted: int = 0

    def __str__(self) -> str:
        return (f"marked {self.expired} expired, "
                f"cleared tracks on {self.tracks_cleared} participants, "
                f"deleted {self.deleted} matches")


def sweep(db, now=None) -> SweepReport:
    now = now or utcnow()
    report = SweepReport()

    # 1. Mark the abandoned ones. get_match() does this lazily when someone
    #    visits, which means a match nobody ever opens again stays `waiting`
    #    forever and keeps its data.
    for match in db.query(Match).filter(Match.status == MatchStatus.WAITING).all():
        if match.is_expired(now):
            match.status = MatchStatus.EXPIRED
            report.expired += 1

    # 2. Drop raw libraries we no longer need. Keeping `n_tracks` and
    #    `n_resolved` because they are the coverage measurement the project
    #    depends on and they identify nobody.
    done = {MatchStatus.COMPLETE, MatchStatus.EXPIRED, MatchStatus.FAILED,
            MatchStatus.INSUFFICIENT, MatchStatus.UNRESOLVED}
    for match in db.query(Match).filter(Match.status.in_(done)).all():
        settled = match.completed_at or match.created_at
        if settled.tzinfo is None:
            from datetime import timezone
            settled = settled.replace(tzinfo=timezone.utc)
        if now - settled < TRACKS_TTL:
            continue
        for p in match.participants:
            if p.track_ids:
                p.track_ids = ""        # not p.tracks = [], which resets n_tracks
                report.tracks_cleared += 1

    # 3. Delete the old ones entirely. Cascades to participants; share cards
    #    are removed explicitly because a dangling token that 404s is fine
    #    but a dangling row is just rot.
    cutoff = now - MATCH_RETENTION
    for match in db.query(Match).all():
        created = match.created_at
        if created.tzinfo is None:
            from datetime import timezone
            created = created.replace(tzinfo=timezone.utc)
        if created > cutoff:
            continue
        db.query(ShareCard).filter(ShareCard.match_id == match.id).delete()
        db.delete(match)
        report.deleted += 1

    db.commit()
    return report


def main() -> None:  # pragma: no cover - entry point
    import os

    from models import make_engine, make_session_factory

    engine = make_engine(os.getenv("DATABASE_URL", "sqlite:///lovify.db"))
    with make_session_factory(engine)() as db:
        print(sweep(db))


if __name__ == "__main__":  # pragma: no cover
    main()
