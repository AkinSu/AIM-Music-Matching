"""Public share cards, and the history page they are made from.

Two links, two jobs. The distinction is the whole module:

    /m/<match_id>        private. A capability to JOIN a match.
    /s/<token>           public.  A capability to READ one finished result.

Conflating them is what made the earlier takeover bug worth exploiting: the
only link in the system was also the link that granted a seat, so it could
never be posted anywhere, and because people obviously *want* to post these,
it would have been posted anyway.

Like matching.py, this is free of Flask so the rules can be tested without a
request context.
"""

from __future__ import annotations

from dataclasses import dataclass

from models import Match, MatchStatus, Participant, ShareCard, utcnow

# Statuses a card may be made from. A match that is still waiting, expired,
# or broken has nothing worth posting, and a share link that renders "we
# could not finish the comparison" is worse than no share link.
SHAREABLE = {MatchStatus.COMPLETE}


class ShareError(Exception):
    """Something the user should be told, in words they can act on."""


# ── my matches ───────────────────────────────────────────────────────────────

@dataclass
class HistoryRow:
    match: Match
    me: Participant
    them: Participant | None
    card: ShareCard | None

    @property
    def shareable(self) -> bool:
        return self.match.status in SHAREABLE

    @property
    def shared(self) -> bool:
        return self.card is not None and self.card.active


def history(db, spotify_id: str, limit: int = 50) -> list[HistoryRow]:
    """Every match this Spotify account is in, newest first.

    Keyed on the Spotify id rather than on the session's per-match role keys:
    cookies get cleared, and a history page that forgets everything the first
    time someone clears their browser is not a history page.
    """
    rows = (db.query(Participant)
              .filter(Participant.spotify_id == spotify_id)
              .all())

    out: list[HistoryRow] = []
    for me in rows:
        match = me.match
        if match is None:
            continue
        them = next((p for p in match.participants if p.id != me.id), None)
        out.append(HistoryRow(match=match, me=me, them=them,
                              card=active_card(db, match.id, spotify_id)))

    out.sort(key=lambda r: r.match.created_at, reverse=True)
    return out[:limit]


# ── share cards ──────────────────────────────────────────────────────────────

def active_card(db, match_id: str, spotify_id: str) -> ShareCard | None:
    return (db.query(ShareCard)
              .filter(ShareCard.match_id == match_id,
                      ShareCard.created_by == spotify_id,
                      ShareCard.revoked_at.is_(None))
              .one_or_none())


def _participant(match: Match, spotify_id: str) -> Participant | None:
    return next((p for p in match.participants if p.spotify_id == spotify_id), None)


def create_card(db, match: Match, spotify_id: str,
                include_details: bool = False) -> ShareCard:
    """Mint a public link for a finished match. Rotates any existing one.

    Rotation rather than reuse is the point Akin asked for, and it is also
    the right default: a link that has escaped into a group chat cannot be
    recalled, so the only recall mechanism a bearer token can have is being
    replaced. Pressing share twice gives you a new link and kills the old.
    """
    if _participant(match, spotify_id) is None:
        raise ShareError("Only the two people in a match can share it.")
    if match.status not in SHAREABLE:
        raise ShareError(
            "This match is not finished, so there is no result to share yet.")

    revoke_mine(db, match, spotify_id)
    card = ShareCard(match_id=match.id, created_by=spotify_id,
                     include_details=include_details)
    db.add(card)
    db.commit()
    return card


def revoke_mine(db, match: Match, spotify_id: str) -> int:
    """Kill my own link, leaving the other person's alone."""
    card = active_card(db, match.id, spotify_id)
    if card is None:
        return 0
    card.revoked_at = utcnow()
    db.commit()
    return 1


def revoke_all(db, match: Match, spotify_id: str) -> int:
    """Kill every public link to this match, including the other person's.

    Either participant may do this, and that asymmetry is deliberate. You can
    publish a result that is half mine; I must be able to unpublish it. A
    veto that only worked on your own links would not be a veto.
    """
    if _participant(match, spotify_id) is None:
        raise ShareError("Only the two people in a match can change its sharing.")

    cards = (db.query(ShareCard)
               .filter(ShareCard.match_id == match.id,
                       ShareCard.revoked_at.is_(None))
               .all())
    for card in cards:
        card.revoked_at = utcnow()
    db.commit()
    return len(cards)


def load_card(db, token: str) -> ShareCard:
    card = db.get(ShareCard, token)
    if card is None or not card.active:
        raise ShareError(
            "This share link is no longer active. Whoever posted it has "
            "replaced it or taken it down.")
    if card.match is None or card.match.status not in SHAREABLE:
        raise ShareError("There is no result behind this link any more.")
    return card


# ── what the public actually sees ────────────────────────────────────────────

def public_view(card: ShareCard) -> dict:
    """The card, reduced to what is safe to put on the open internet.

    Built by naming what goes in, never by deleting from the full result.
    A view model that starts from everything and removes the private bits
    leaks every field anyone adds later; this one leaks nothing by default,
    which is the only way round that survives other people editing the code.
    """
    import json

    match = card.match
    result = json.loads(match.result_json) if match.result_json else {}

    inviter, invited = match.inviter, match.invited
    names = [p.display_name or "Someone" for p in (inviter, invited) if p]

    view = {
        "token": card.token,
        "names": names,
        "percentile": result.get("percentile"),
        "interpretation": result.get("interpretation"),
        "completed_at": match.completed_at,
        "include_details": card.include_details,
        # Deliberately absent unless opted in: shared_artists, genres,
        # components, track counts, photos, Spotify ids, the match id.
        "shared_artists": [],
        "top_shared_genres": [],
    }

    if card.include_details:
        view["shared_artists"] = result.get("shared_artists", [])[:8]
        view["top_shared_genres"] = result.get("top_shared_genres", [])[:5]

    return view


def share_text(view: dict) -> str:
    """One line for the socials. Short enough for X without a link."""
    names = view.get("names") or []
    who = " and ".join(names) if len(names) == 2 else "We"
    pct = view.get("percentile")
    if pct is None:
        return f"{who} compared our music taste on Lovify."
    return (f"{who} landed in the {pct:g}th percentile for music taste "
            f"compatibility on Lovify.")
