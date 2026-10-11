#!/usr/bin/env python
"""Check the notebooks in CI, where the real data does not exist.

Two different checks, because the notebooks are two different kinds of thing.

**Most of them go through `data.load_*`**, which falls back to synthetic data
when `data/raw/` is empty. Those can be re-executed from scratch on a clean
runner, and that is the check that matters: a notebook which has silently
stopped running is how analysis rots.

**Some read `data/raw/` directly** -- notebook 01 reads the real scrape,
because its whole subject is what the real scrape says. `data/raw/` is
gitignored (the catalog alone is 100+ MB, and the scrape is ours to regenerate
rather than commit), so those cannot be re-executed here. Pretending otherwise
is what broke the first version of this job.

For those we verify the committed artifact instead: every cell parses, no cell
carries an error output, and execution counts are in order. That catches a
broken or half-run notebook without pretending we can reproduce an analysis
whose inputs are not present.

The split is detected, not hardcoded, so adding a notebook does not require
remembering to update a list here.

    cd ml && python scripts/ci_notebooks.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# A direct read of data/raw or data/interim means the notebook needs inputs
# that are not in the repository.
NEEDS_REAL = re.compile(r"data/(raw|interim)/")


def structural_check(path: Path) -> list[str]:
    """Verify the committed notebook is coherent, without running it."""
    problems: list[str] = []
    nb = nbformat.read(path, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            problems.append(f"cell {i} does not parse: {exc.msg}")
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                problems.append(f"cell {i} has an error output: "
                                f"{out.get('ename')}: {out.get('evalue')}")

    counts = [c.execution_count for c in nb.cells
              if c.cell_type == "code" and c.execution_count is not None]
    if counts != sorted(counts):
        problems.append("execution counts are out of order -- was it re-run top to bottom?")
    if not counts:
        problems.append("no cell has been executed; outputs are not committed")

    return problems


def main() -> int:
    executable, artifact_only = [], []
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        source = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        (artifact_only if NEEDS_REAL.search(source) else executable).append(path)

    failures = 0

    print("── structural check (all notebooks) " + "─" * 36)
    for path in NOTEBOOKS:
        problems = structural_check(path)
        if problems:
            failures += 1
            print(f"  FAIL  {path.name}")
            for p in problems:
                print(f"          {p}")
        else:
            print(f"  ok    {path.name}")

    print("\n── execution check (synthetic-capable only) " + "─" * 28)
    for path in artifact_only:
        print(f"  skip  {path.name}  (reads data/raw directly; not in the repo)")

    for path in executable:
        out = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--ExecutePreprocessor.timeout=1800",
             "--output-dir", "/tmp/nb-ci", str(path)],
            capture_output=True, text=True, cwd=ROOT,
        )
        if out.returncode == 0:
            print(f"  ok    {path.name}")
        else:
            failures += 1
            print(f"  FAIL  {path.name}")
            tail = (out.stderr or out.stdout).strip().splitlines()[-12:]
            for line in tail:
                print(f"          {line}")

    print()
    if failures:
        print(f"{failures} notebook check(s) failed")
        return 1
    print(f"all {len(NOTEBOOKS)} notebooks pass "
          f"({len(executable)} executed, {len(artifact_only)} structure-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
