"""Database models for the cross-device match flow.

The app used to keep both participants in `session[...]` -- one browser's
cookie. That silently assumed the second person was sitting in the first
person's browser. A real invitee on a real phone gets a fresh session with no
trace of the inviter, so the results page had half the data and no way to get
the other half.

Everything shared between two people therefore lives here, server side, keyed
by an id that travels in the share link. The cookie session is reduced to one
job: remembering which participant *this browser* is.

Note on what is deliberately absent: there is no column for a Spotify access
or refresh token. Tokens are used once during the OAuth callback to fetch the
user's top tracks and are then dropped on the floor. Storing them would make
this database worth stealing and would drag in the whole token-refresh
problem; we need the data once, so we take it once.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)

# How long an invitation stays open. Long enough to text someone and have
# them get to it after dinner; short enough that abandoned half-matches, which
# hold one person's listening data, do not accumulate indefinitely.
MATCH_TTL = timedelta(days=7)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_match_id() -> str:
    """Unguessable id for a match.

    This id IS the share link, so it is a bearer capability: anyone holding
    it can join the match. 32 hex chars from `secrets` makes guessing one
    infeasible, which is the only protection a shareable link can have.
    """
    return secrets.token_hex(16)


class Base(DeclarativeBase):
    pass


class MatchStatus:
    """Status values the templates switch on.

    Strings rather than an enum so they survive a round trip through JSON and
    a database without ceremony, and so a new state can be added without a
    migration.
    """

    WAITING = "waiting"            # inviter joined, waiting on the invitee
    COMPLETE = "complete"          # both joined, result computed
    INSUFFICIENT = "insufficient"  # someone had too little listening history
    UNRESOLVED = "unresolved"      # someone's music is not in our feature store
    EXPIRED = "expired"
    FAILED = "failed"              # scoring itself broke; distinct from the above

    # The two states below are outcomes, not errors. ml's compatibility()
    # returns them by design when it cannot honestly produce a number, and the
    # UI has a screen for each.
    NON_ERROR_OUTCOMES = {WAITING, COMPLETE, INSUFFICIENT, UNRESOLVED}


class Match(Base):
    """One invitation, and the comparison it eventually produces."""

    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_match_id)
    status: Mapped[str] = mapped_column(String(20), default=MatchStatus.WAITING,
                                        index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: utcnow() + MATCH_TTL)

    # The scored result, as JSON text. Stored rather than recomputed: the two
    # participants load it at different times, and recomputing would mean
    # keeping their track lists alive longer than necessary.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Why a match ended up INSUFFICIENT / UNRESOLVED / FAILED, in words the UI
    # can show without having to re-derive them.
    status_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    participants: Mapped[list["Participant"]] = relationship(
        back_populates="match", cascade="all, delete-orphan")

    # ── helpers ──────────────────────────────────────────────────────────────
    def participant(self, role: str) -> "Participant | None":
        return next((p for p in self.participants if p.role == role), None)

    @property
    def inviter(self) -> "Participant | None":
        return self.participant(Role.INVITER)

    @property
    def invited(self) -> "Participant | None":
        return self.participant(Role.INVITED)

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or utcnow()
        expires = self.expires_at
        # SQLite hands back naive datetimes even for timezone=True columns.
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Match {self.id[:8]} {self.status} n={len(self.participants)}>"


class Role:
    INVITER = "inviter"
    INVITED = "invited"


class Participant(Base):
    """One person inside one match.

    `spotify_id` is kept so a person who opens their own share link is
    recognised rather than being matched against themselves.
    """

    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"),
                                          index=True)
    role: Mapped[str] = mapped_column(String(10))

    spotify_id: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(200), default="")
    photo_url: Mapped[str] = mapped_column(String(500), default="")

    # Their top tracks, as a newline-separated list of Spotify track ids.
    # The derived data we actually need -- not the token that produced it.
    track_ids: Mapped[str] = mapped_column(Text, default="")
    n_tracks: Mapped[int] = mapped_column(Integer, default=0)

    # How many of those tracks we could actually find audio features for, and
    # where the library came from. Nobody has measured either for a real
    # Spotify user -- coverage is the single biggest unknown in the design,
    # and whether saved songs are enough to clear MIN_LIBRARY_TRACKS decides
    # whether the product works for ordinary people. Recorded on the row so
    # the first real match answers it instead of leaving us guessing.
    n_resolved: Mapped[int] = mapped_column(Integer, default=0)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                default=utcnow)

    match: Mapped[Match] = relationship(back_populates="participants")

    __table_args__ = (
        # One inviter and one invited per match, enforced by the database
        # rather than by hoping the application never double-writes.
        Index("ux_participant_match_role", "match_id", "role", unique=True),
    )

    @property
    def coverage(self) -> float:
        """Share of this person's tracks with audio features. 0.0 when unknown."""
        return (self.n_resolved / self.n_tracks) if self.n_tracks else 0.0

    @property
    def tracks(self) -> list[str]:
        return [t for t in self.track_ids.split("\n") if t]

    @tracks.setter
    def tracks(self, ids: list[str]) -> None:
        clean = [t for t in dict.fromkeys(ids) if t]
        self.track_ids = "\n".join(clean)
        self.n_tracks = len(clean)


class TrackFeature(Base):
    """Audio features for one track, looked up by Spotify track id.

    This is the feature store -- 1.4M rows -- and it is a table rather than a
    DataFrame held in memory for one reason: every worker process would
    otherwise carry its own ~200MB copy, while an indexed lookup on a few
    hundred ids is a single fast query and costs no resident memory.

    It exists at all because Spotify removed /v1/audio-features on
    2024-11-27. These numbers cannot be fetched live any more, at any price.
    """

    __tablename__ = "track_features"

    track_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    danceability: Mapped[float] = mapped_column(Float)
    energy: Mapped[float] = mapped_column(Float)
    speechiness: Mapped[float] = mapped_column(Float)
    acousticness: Mapped[float] = mapped_column(Float)
    instrumentalness: Mapped[float] = mapped_column(Float)
    liveness: Mapped[float] = mapped_column(Float)
    valence: Mapped[float] = mapped_column(Float)
    tempo: Mapped[float] = mapped_column(Float)
    loudness: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[float] = mapped_column(Float)

    # Carried when the source has them; the similarity blend uses genre and
    # year, and they are absent from some feature dumps.
    genre: Mapped[str | None] = mapped_column(String(120), nullable=True)
    artist: Mapped[str | None] = mapped_column(String(300), nullable=True)
    popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    AUDIO_COLUMNS = ("danceability", "energy", "speechiness", "acousticness",
                     "instrumentalness", "liveness", "valence", "tempo",
                     "loudness", "duration_ms")

    def as_row(self) -> dict:
        row = {c: getattr(self, c) for c in self.AUDIO_COLUMNS}
        row.update(track_id=self.track_id, genre=self.genre or "unknown",
                   artist=self.artist or "unknown",
                   popularity=self.popularity if self.popularity is not None else 50,
                   year=self.year if self.year is not None else 2015)
        return row


# ── engine / session ─────────────────────────────────────────────────────────

def normalise_database_url(database_url: str) -> str:
    """Make a hosting provider's DATABASE_URL something SQLAlchemy can open.

    Two separate corrections, both of which otherwise fail only in
    deployment, where they are expensive to debug:

    1. Railway (and Heroku) inject a URL beginning `postgres://`, a scheme
       SQLAlchemy 2 no longer recognises.

    2. A bare `postgresql://` means psycopg2 to SQLAlchemy, but the image is
       built with psycopg 3, so the import fails at startup. Naming the
       driver that is actually installed keeps the app working under either
       requirements file rather than only the one it was tested against.
    """
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    if database_url.startswith("postgresql://"):
        from importlib.util import find_spec
        if find_spec("psycopg2") is None and find_spec("psycopg") is not None:
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1)
    return database_url


def make_engine(database_url: str, echo: bool = False):
    """Engine for SQLite locally and Postgres in deployment."""
    database_url = normalise_database_url(database_url)

    kwargs: dict = {"echo": echo, "future": True}
    if database_url.startswith("sqlite"):
        # check_same_thread is a SQLite-only guard that Flask's threaded dev
        # server trips over; harmless to relax for a single-process app.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
