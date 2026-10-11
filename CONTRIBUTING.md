# How we work

Several contributors on one repo, some of them human and some of them agents. This is
the smallest process that stops them overwriting each other.

The rules below are written to apply to both. An agent gets handed a ticket, works on
its own branch, and opens a PR like anyone else; the review gate is what keeps its work
honest, and in this project that gate has already caught a 62-point error in a result
that four other agents had read past. Nothing here is ceremony — every rule exists
because something went wrong without it.

## The loop

```
Linear ticket  ->  branch  ->  PR  ->  review  ->  merge  ->  ticket auto-closes
```

Nothing goes onto `main` directly. Not a typo fix, not a README tweak.

## 1. Pick up a ticket

Every piece of work starts as a Linear issue. If you're doing something that isn't a
ticket, make the ticket first — it takes fifteen seconds and it's how the rest of the
team knows what you're on.

Move it to **In Progress** and assign it to yourself.

## 2. Branch

In Linear, with the issue open, press **Cmd+Shift+.** to copy the generated branch name.
It looks like:

```
akin/lov-14-cluster-near-duplicate-playlists
```

Then:

```bash
git checkout main
git pull
git checkout -b akin/lov-14-cluster-near-duplicate-playlists
```

**Use the name Linear gives you.** The `lov-14` in it is what links the branch, its PR,
and the ticket together — the GitHub integration keys off that string. Rename the
branch and the automation silently stops working.

## 3. Commit as you go

Small commits with real messages. `fix stuff` tells the next person nothing.

```bash
git add ml/src/lovify_ml/dedup.py
git commit -m "Cluster playlists by Jaccard over track sets"
```

## 4. Open a PR

```bash
git push -u origin HEAD
gh pr create --fill
```

Put `LOV-14` in the PR title or description and Linear links them automatically. Opening
the PR moves the ticket to **In Review**; merging moves it to **Done**.

## 5. Get a review

**Every PR needs one approval from someone who didn't write it** — and when the author
was an agent, the reviewer must not be the same agent. Self-review by the thing that
produced the work catches almost nothing; independent review on this project has
already caught a withdrawn headline result and a live scoring bug.

When you review, actually pull the branch and run it. "LGTM" on a diff you didn't
execute is not a review.

What to look for:

- **Does the claim match the output?** If a notebook says "accuracy improved" the cell
  below it must show that. This is where our worst bug lived.
- **Any chance of leakage?** Does the same track, user, or playlist appear on both sides
  of a split?
- **Is the baseline there?** A number with no chance rate or random baseline next to it
  is not interpretable.
- **Does it run from a clean checkout?** No absolute paths, no "works on my machine"
  dependency.

## 6. Merge

Squash and merge. Delete the branch. Linear closes the ticket.

---

## Branch naming

| prefix | for |
|---|---|
| `<name>/lov-N-slug` | normal work, from Linear |
| `hotfix/<slug>` | the app is broken in production |

## What never goes in a commit

- `.env`, client secrets, access tokens, refresh tokens
- `data/raw/` contents — the Kaggle dump is 600k rows, keep it local
- `ml/artifacts/` — trained models, regenerate them instead
- `.DS_Store`

All of these are gitignored. If you find yourself using `git add -f`, stop.

## Notebooks

Notebooks are committed **with outputs**, so the team and anyone reading the repo can
see the results without running anything. That means:

- Re-run the whole notebook top to bottom before committing. A notebook with
  out-of-order execution counts is not reviewable.
- If you only changed prose, don't re-execute — that creates a huge diff of unchanged
  image data for no reason.
- Never commit a notebook containing a credential in a cell output.

## Running things

```bash
cd ml
.venv/bin/python -m pytest tests/ -q          # tests
.venv/bin/jupyter lab notebooks/              # notebooks
.venv/bin/python scripts/check_spotify_access.py   # what the API still allows
```
