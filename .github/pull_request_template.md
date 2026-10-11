## What and why

<!-- One or two sentences. What changed, and what problem it solves. -->

Closes LOV-

## How I verified it

<!-- Commands you ran and what came back. "Tests pass" is not enough on its own -->
<!-- if the change affects a model: paste the numbers. -->

```
```

## Checklist

- [ ] Runs from a clean checkout (no absolute paths, no uncommitted local files)
- [ ] `cd ml && .venv/bin/python -m pytest tests/ -q` passes
- [ ] No credentials, tokens, or `data/raw/` contents in the diff

If this touches a model or a notebook:

- [ ] Every claim in the prose is backed by an output visible in the same notebook
- [ ] Any train/test split is grouped so the same track, user or playlist cannot
      appear on both sides
- [ ] Results are reported against a baseline (chance rate, or random recommendation)
- [ ] Notebook was re-run top to bottom, so execution counts are in order
