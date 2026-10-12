# Security

What this app does to protect the people using it, what it deliberately does
not do, and how to check any of it yourself.

Scope: one Flask app, a Postgres database, and Spotify OAuth. No payments,
no passwords of our own, no health data. What we hold that matters is **a
list of the music someone listens to, linked to their name and face** —
which is personal, inferable (mood, age, politics, religion), and not
something anyone expects to leak out of a toy compatibility app.

---

## The two links

This is the design decision most of the rest follows from.

| | invite link | share link |
|---|---|---|
| URL | `/m/<match_id>` | `/s/<token>` |
| What it grants | a **seat** in a match | **read** one finished result |
| Audience | one person you texted | anyone |
| Revocable | no (expires in 7 days) | yes, instantly, by either participant |
| Created by | starting a match | pressing a button |
| Shows the result | only to the two participants | yes, by design |

They used to be one link, and that was the problem. The only URL in the
system both granted a seat *and* displayed the result, so it could never be
posted anywhere — while being exactly the kind of thing people post.

Two separate failures came out of that:

1. **Seat takeover.** `join_match` overwrote a participant unconditionally.
   Anyone with the invite link could evict a participant after the match
   completed and be re-scored in their place, repeatedly, using the
   "artists you both love" list as an oracle. Reproduced: a handful of
   probes from one account recovered a victim's library exactly. Fixed by
   binding a seat to the first `spotify_id` that claims it
   (`matching.py`, `tests/test_takeover.py`).
2. **Unintended publication.** A bystander with the link saw the result.
   Nobody chose that. Results are participants-only now, and publishing is
   a separate token someone has to create (`sharing.py`).

---

## Transport

| | |
|---|---|
| HTTPS | forced. Plain http gets a 301; `X-Forwarded-Proto` is honoured so a TLS-terminating proxy does not loop |
| HSTS | `max-age=31536000; includeSubDomains` — a year, preload-eligible |
| Local dev | **no HSTS**. An HSTS header from `127.0.0.1` pins that host to https in the developer's browser for a year and breaks every other local project they own |

Certificate and cipher configuration belong to the host (Railway, Fly). Check
the deployed origin at [SSL Labs](https://www.ssllabs.com/ssltest/) before
launch; nothing in this repository can get an A and nothing in it can stop
you getting an F.

## Response headers

Every one of these is asserted in `tests/test_security_headers.py`, because a
security header is exactly the kind of thing that gets added once and then
silently dropped by a refactor.

| header | value | what it closes |
|---|---|---|
| `Content-Security-Policy` | see below | injected script |
| `X-Frame-Options` / `frame-ancestors` | `DENY` / `'none'` | clickjacking — a share card is exactly what you would frame invisibly |
| `X-Content-Type-Options` | `nosniff` | MIME confusion |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | the share URL is a bearer token and the card links out to X and Facebook; without this the full token travels to them in `Referer` |
| `Permissions-Policy` | geolocation, camera, mic, payment, usb all `()` | features this app has no business asking for |
| `Cross-Origin-Opener-Policy` | `same-origin` | cross-window tampering |

**`script-src` carries no `'unsafe-inline'`.** That is the directive that
actually stops an injected `<script>`, so nothing in this project may use an
inline handler or an inline script block; `static/lovify.js` exists because
of that rule. A test walks every template and fails on an `onclick` or a
bare `<script>`, and another checks that every external URL in the templates
is permitted by the policy — which caught a real mistake, cdnjs being
allowed for scripts when it actually serves a stylesheet.

`style-src` does carry `'unsafe-inline'`, for one reason: the component bars
on the result page set a computed width with a `style` attribute. Inline
style on a page with no inline script is a far better trade than generating
a stylesheet per request.

## Cookies, and what is in them

One cookie, `lovify_session`, signed with `SECRET_KEY`.

- `HttpOnly` — script cannot read it, so an XSS cannot steal a session
- `SameSite=Lax` — it does not ride along on cross-site requests
- `Secure` — set whenever `BASE_URL` is not local

It holds two kinds of key and nothing else: `role:<match_id>` (which seat
this browser has, per match) and `spotify_id` (who this browser is, for the
history page). A test asserts the exact key set, because the bug this app
was rewritten to fix was *the inviter's data living in the invitee's
cookie*.

**The public card sets no cookie at all.** Someone opening a shared link
from a group chat is not a user of ours.

## CSRF

The two state-changing routes (`/m/<id>/share`, `/m/<id>/unshare`) require a
signed token, compared in constant time. `SameSite=Lax` already blocks the
cross-site form post this guards against in any browser that honours it; the
token is the layer that does not depend on the browser getting that right.

Without it, any page Bob visited could silently unpublish his match — or
publish it.

## What we store, and for how long

Deliberately absent from the database: **no Spotify access token, no refresh
token, no password.** Tokens are used once in the OAuth callback to read the
library and are then dropped. That removes the token-refresh problem and
makes this database much less worth stealing.

`retention.py` (`make sweep`, daily in deployment):

| age | what happens |
|---|---|
| 7 days, never joined | marked expired |
| 1 day after scoring | **raw track lists deleted.** The result is already computed and stored; the library was only ever an input |
| 180 days | match deleted entirely, share cards included |

`n_tracks` and `n_resolved` survive the sweep. They are the feature-coverage
measurement the project depends on and they identify nobody.

## Third parties

Every external host the browser contacts, and why:

| host | what for | sees |
|---|---|---|
| `accounts.spotify.com` | OAuth | that you logged in (they are the identity provider) |
| `i.scdn.co`, `*.fbcdn.net` | profile pictures | your IP, when a picture loads |
| `fonts.googleapis.com` / `gstatic.com` | the Nunito webfont | **your IP, on every page load** |
| `cdnjs.cloudflare.com` | Font Awesome icons | your IP, on the older pages |

No analytics, no ad network, no tracking pixel, no third-party cookie.

**Open item: the Google Fonts request is a real IP leak** to a company whose
business is advertising, on every page view, including the public share card
that strangers open. It is not a tracker in the cookie sense and it is very
common, but it is also entirely avoidable: self-host the two `.woff2` files
and drop both the `fonts.googleapis.com` and `fonts.gstatic.com` exceptions
from the CSP. Worth doing before any real launch. Same argument applies to
cdnjs on the legacy pages.

## Logging

Match ids and share tokens are **bearer capabilities** — anyone who reads
one out of a log file can use it. They are logged as a 12-character SHA-256
prefix (`security.digest`), which stays traceable through a log and is no
longer usable from one. A test asserts the raw id never appears.

## What this does not do

Named, so nobody assumes otherwise:

- **No rate limiting.** The Spotify app is in development mode, capped at 25
  allowlisted users. A blocklist we cannot test is worse than none.
- **No account system.** There is no password to steal because there is no
  account. Identity is a cookie carrying a Spotify id, which is appropriate
  for listing your own matches and is not a security boundary for anything
  destructive beyond your own share links.
- **No audit log.** Nobody is accountable to anybody yet.
- **No encryption at rest** beyond whatever the database host provides.
- **Spotify profile names and photos are shown as given.** Jinja escapes
  them, and the CSP restricts image hosts, so a hostile display name cannot
  execute — but it can still be rude.

## Checking it yourself

```bash
# headers, as deployed
curl -sI https://<your-host>/ | grep -iE 'content-security|strict-transport|x-frame|referrer|permissions'

# the whole security suite
cd "LOVIFY ME APP" && SECRET_KEY=x python -m pytest tests/test_security_headers.py -v

# TLS configuration        https://www.ssllabs.com/ssltest/
# headers, graded          https://securityheaders.com/
# malware and blacklists   https://sitecheck.sucuri.net/
```

The external scanners only work against a deployed origin, not against
`localhost`.

## Reporting something

Open a GitHub issue, or tell Akin directly if you would rather it were not
public first.
