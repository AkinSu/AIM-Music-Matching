"""Match lifecycle: create, join, score.

Deliberately free of Flask. Everything here takes a database session and
plain arguments, so the rules can be tested without spinning up a web server
or faking a request context -- and so the one behaviour that actually matters,
two *separate* browsers completing a match, can be exercised directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from models import Match, MatchStatus, Participant, Role, utcnow


class MatchError(Exception):
    """Something the user should be told, in words they can act on."""


@dataclass
class JoinResult:
    match: Match
    participant: Participant
    scored: bool


# ── create ───────────────────────────────────────────────────────────────────

def create_match(db) -> Match:
    """Open an invitation. The inviter attaches on their OAuth return."""
    match = Match()
    db.add(match)
    db.commit()
    return match


def get_match(db, match_id: str) -> Match:
    match = db.get(Match, match_id)
    if match is None:
        raise MatchError("That link does not point at a match. It may have been mistyped.")
    if match.is_expired() and match.status != MatchStatus.COMPLETE:
        if match.status != MatchStatus.EXPIRED:
            match.status = MatchStatus.EXPIRED
            db.commit()
        raise MatchError(
            "This invitation has expired. Ask for a fresh link and try again.")
    return match


# ── join ─────────────────────────────────────────────────────────────────────

def join_match(db, match: Match, role: str, profile, track_ids: list[str],
               scorer=None, sources: dict | None = None) -> JoinResult:
    """Attach a person to a match, and score it once both sides are in.

    Re-joining in the same role is allowed and overwrites: people refresh,
    lose signal mid-redirect, and retry. Making that an error would strand
    them with no way forward.
    """
    if role not in (Role.INVITER, Role.INVITED):
        raise MatchError("Unknown role for this invitation.")

    other = match.participant(Role.INVITED if role == Role.INVITER else Role.INVITER)
    if other is not None and other.spotify_id == profile.spotify_id:
        raise MatchError(
            "That is the same Spotify account as the other person in this match. "
            "Open the link on a different account to compare two people.")

    participant = match.participant(role)
    if participant is None:
        # First claim on this role. Whoever lands here owns it from now on.
        participant = Participant(match_id=match.id, role=role,
                                  spotify_id=profile.spotify_id)
        db.add(participant)
        match.participants.append(participant)
    elif participant.spotify_id != profile.spotify_id:
        # A DIFFERENT Spotify account trying to take an occupied seat.
        #
        # This used to overwrite, which made the share link far more powerful
        # than intended: anyone holding it could evict a participant -- after
        # the match was complete -- and re-score in their place. Repeatedly.
        #
        # That turned the results page into an oracle. Each takeover asks
        # "does the other person listen to X?" and `shared_artists` answers.
        # Reproduced: six probes recovered a victim's library exactly, from
        # one reused account and nothing but the link. For a product about
        # comparing music with someone you like, that is the attack.
        #
        # The retry case the old comment protected -- lost signal mid-redirect,
        # a refresh -- is the branch below: the SAME account re-joining still
        # overwrites its own row freely.
        raise MatchError(
            "Someone has already joined this invitation. Each link compares "
            "exactly two people. Ask them to send you a new one.")

    participant.spotify_id = profile.spotify_id
    participant.display_name = profile.display_name
    participant.photo_url = profile.photo_url
    participant.tracks = track_ids
    participant.sources_json = json.dumps(sources) if sources else None
    participant.joined_at = utcnow()
    db.commit()

    scored = False
    if match.inviter is not None and match.invited is not None:
        score_match(db, match, scorer)
        scored = True

    return JoinResult(match=match, participant=participant, scored=scored)


# ── score ────────────────────────────────────────────────────────────────────

def score_match(db, match: Match, scorer=None) -> dict:
    """Compute and persist the result.

    Stored rather than recomputed on each view: the two people open the page
    at different times, and keeping the result means their track lists do not
    have to stay around waiting for the second visit.
    """
    from scoring import features_for

    a, b = match.inviter, match.invited
    if a is None or b is None:
        raise MatchError("Both people have to join before there is anything to compare.")

    if scorer is None or not getattr(scorer, "ready", False):
        detail = getattr(getattr(scorer, "unavailable", None), "reason",
                         "No scorer was supplied.")
        return _finish(db, match, MatchStatus.FAILED, {
            "status": "scoring_unavailable",
            "percentile": None,
            "interpretation": "Compatibility scoring is not available on this server yet.",
        }, detail)

    frame_a = features_for(db, a.tracks)
    frame_b = features_for(db, b.tracks)

    # The number nobody has measured for a real Spotify user: how much of a
    # person's library we can actually find audio features for. Persisted so
    # the first real match tells us whether the fetch strategy is sufficient.
    a.n_resolved, b.n_resolved = len(frame_a), len(frame_b)
    db.commit()

    # Nothing resolvable is its own outcome. It happens for real people whose
    # listening is too new or too obscure for a pre-2021 feature dump, and
    # telling them that is far better than a confident wrong number.
    empty = [p.display_name or "they" for p, f in ((a, frame_a), (b, frame_b)) if not len(f)]
    if empty:
        return _finish(db, match, MatchStatus.UNRESOLVED, {
            "status": "no_resolvable_tracks",
            "percentile": None,
            "interpretation": (
                f"We could not find audio data for {' or '.join(empty)}'s music. "
                "That usually means it is too new or too obscure for our database."),
        }, f"unresolved for: {', '.join(empty)}")

    result = scorer.compatibility(frame_a, frame_b)
    status = {
        "ok": MatchStatus.COMPLETE,
        "insufficient_library": MatchStatus.INSUFFICIENT,
        "no_resolvable_tracks": MatchStatus.UNRESOLVED,
    }.get(result.get("status"), MatchStatus.FAILED)

    return _finish(db, match, status, result, result.get("detail"))


def _finish(db, match: Match, status: str, result: dict, detail: str | None) -> dict:
    match.status = status
    match.result_json = json.dumps(result, default=str)
    match.status_detail = detail
    match.completed_at = utcnow()
    db.commit()
    return result


def result_for(match: Match) -> dict | None:
    if not match.result_json:
        return None
    return json.loads(match.result_json)


def view_model(match: Match, viewer_role: str | None) -> dict:
    """What the results template needs, from the viewer's point of view.

    Each person should see the other person named as the other person, not as
    "user two", so the labels depend on who is looking.
    """
    inviter, invited = match.inviter, match.invited
    me, them = (inviter, invited) if viewer_role == Role.INVITER else (invited, inviter)
    if viewer_role is None:
        me, them = inviter, invited

    def card(p: Participant | None) -> dict:
        if p is None:
            return {}
        return {"user_name": p.display_name, "user_photo": p.photo_url,
                "n_tracks": p.n_tracks}

    return {
        "match_id": match.id,
        "status": match.status,
        "you": card(me),
        "them": card(them),
        "result": result_for(match),
        "is_participant": viewer_role is not None,
        "waiting_on_other": match.status == MatchStatus.WAITING,
    }
