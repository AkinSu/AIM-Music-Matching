#!/usr/bin/env python
"""Load the audio-feature store into Postgres.

Why a table and not a file in memory: the store is 1.4M rows and ~135MB on
disk. Read into pandas it costs roughly 200MB of RSS *per gunicorn worker*,
and the app only ever needs point lookups -- "what does this one track sound
like" -- which an indexed table answers in well under a millisecond at no
standing memory cost.

    make seed          # normal use
    python scripts/seed_db.py --help

This script does NOT define the schema. `track_features` is owned by
LOVIFY ME APP/models.py and created by init_db() when the app starts; the
seeder reads the live table definition and fills it. That boundary matters --
an earlier version of this file created its own table and silently disagreed
with the app's, which is the kind of split-brain that only shows up in
production.

Two sources, because they carry different things:

    feature_store.csv  1.42M rows, audio features only
    tracks.csv           587k rows, audio features PLUS genre/artist/year

The audio columns come from the feature store, which has the widest coverage.
Then genre, artist, popularity and year are filled in from the catalog where
it overlaps -- the similarity blend uses genre and year, and models.py marks
them nullable precisely because not every dump has them.

Safe to run repeatedly and safe to interrupt. Rows are inserted with ON
CONFLICT DO NOTHING and committed per chunk, so a second run tops up what is
missing rather than starting over or duplicating.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    import psycopg
except ImportError:
    sys.exit("psycopg is not installed. Run this inside the container: make seed")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "ml" / "data" / "raw"

# The audio columns models.py declares non-nullable. Anything outside this set
# in the CSV (key, mode, time_signature) the app does not model, so it is
# dropped rather than forced into a column that does not exist.
AUDIO = ["danceability", "energy", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "loudness",
         "duration_ms"]
ENRICH = ["genre", "artist", "popularity", "year"]


def normalise_dsn(url: str) -> str:
    """SQLAlchemy-style URLs are what the app and Railway use; psycopg wants plain."""
    return url.replace("postgresql+psycopg://", "postgresql://") \
              .replace("postgres://", "postgresql://")


def table_columns(cur) -> dict[str, int | None]:
    """Column name -> max length for varchar columns, None for the rest.

    Read from the live table rather than hardcoded, so that widening a column
    in models.py does not silently keep truncating here.
    """
    cur.execute("""
        SELECT column_name, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'track_features'
    """)
    return {r[0]: r[1] for r in cur.fetchall()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=RAW / "feature_store.csv")
    ap.add_argument("--catalog", type=Path, default=RAW / "tracks.csv",
                    help="optional, supplies genre/artist/popularity/year")
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--skip-enrich", action="store_true")
    args = ap.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL is not set. Inside the stack it is provided for "
                 "you; outside, export it or pass --database-url.")

    # Both CSVs are gitignored and large, so a fresh clone will not have them.
    # That is expected, not an error: say what to do and exit clean, so a
    # `make fresh` does not abort for someone who has not built them yet.
    if not args.features.exists():
        print(f"No feature store at {args.features}")
        print()
        print("  Gitignored (135MB), so a fresh clone will not have it. Build it:")
        print("      cd ml && .venv/bin/python scripts/prepare_catalog.py")
        print("  which needs the Kaggle dump unzipped into ml/data/raw/kaggle_600k/.")
        print("  See ml/README.md and docs/LOCAL_SETUP.md.")
        print()
        print("Skipping seed. The app still runs; track lookups just return nothing.")
        return 0

    started = time.time()

    with psycopg.connect(normalise_dsn(args.database_url), autocommit=False) as conn:
        with conn.cursor() as cur:
            cols = table_columns(cur)

        if not cols:
            print("Table track_features does not exist yet.")
            print()
            print("  It is created by the app at startup (models.py / init_db).")
            print("  Start the stack first:  make up")
            return 1

        missing = [c for c in AUDIO if c not in cols]  # cols is now a dict
        if missing:
            sys.exit(f"track_features is missing expected columns {missing}. "
                     "models.py and this script have diverged.")

        enrich_cols = [] if args.skip_enrich else [c for c in ENRICH if c in cols]

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM track_features")
            already = cur.fetchone()[0]
        if already:
            print(f"track_features already holds {already:,} rows; topping up.")

        read, inserted = _load_features(conn, args.features, args.chunk)

        updated = 0
        if enrich_cols and args.catalog.exists():
            print(f"\nenriching {', '.join(enrich_cols)} from {args.catalog.name}…")
            updated = _enrich(conn, args.catalog, enrich_cols, args.chunk, cols)
        elif enrich_cols:
            print(f"\nno catalog at {args.catalog.name}; "
                  f"leaving {', '.join(enrich_cols)} null")

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM track_features")
            total = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM track_features WHERE genre IS NOT NULL")
            with_genre = cur.fetchone()[0] if "genre" in cols else 0
        conn.commit()

    elapsed = time.time() - started
    print()
    print(f"read     {read:,} rows from {args.features.name}")
    print(f"inserted {inserted:,} new ({read - inserted:,} already present)")
    if updated:
        print(f"enriched {updated:,} rows with genre/year")
    for col, n in sorted(TRUNCATED.items()):
        print(f"  note: {n:,} {col} values clipped to "
              f"{cols.get(col)} chars to fit the schema")
    print(f"total    {total:,} rows, {with_genre:,} with a genre")
    print(f"took     {elapsed:.1f}s")
    return 0


def _load_features(conn, path: Path, chunk_size: int) -> tuple[int, int]:
    """COPY the audio columns in, chunk by chunk, committing as we go."""
    with conn.cursor() as cur:
        # UNLOGGED landing strip: never read by the app, meaningless after the
        # merge, so skipping the WAL is a large win on 1.4M rows.
        cur.execute("""
            CREATE UNLOGGED TABLE IF NOT EXISTS _seed_staging (
                track_id text PRIMARY KEY,
                danceability double precision, energy double precision,
                speechiness double precision, acousticness double precision,
                instrumentalness double precision, liveness double precision,
                valence double precision, tempo double precision,
                loudness double precision, duration_ms double precision
            );
            TRUNCATE _seed_staging;
        """)
    conn.commit()

    cols = ["track_id"] + AUDIO
    copy_sql = f"COPY _seed_staging ({','.join(cols)}) FROM STDIN"
    merge_sql = (f"INSERT INTO track_features ({','.join(cols)}) "
                 f"SELECT {','.join(cols)} FROM _seed_staging "
                 f"ON CONFLICT (track_id) DO NOTHING")

    read = inserted = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for col in cols:
            if col not in reader.fieldnames:
                sys.exit(f"{path.name} has no '{col}' column. Got: {reader.fieldnames}")

        batch: list[tuple] = []
        for row in reader:
            batch.append(tuple(row[c] or None for c in cols))
            if len(batch) >= chunk_size:
                inserted += _flush(conn, batch, copy_sql, merge_sql, "_seed_staging")
                read += len(batch)
                batch.clear()
                print(f"  {read:>9,} read  {inserted:>9,} new", flush=True)
        if batch:
            inserted += _flush(conn, batch, copy_sql, merge_sql, "_seed_staging")
            read += len(batch)

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _seed_staging")
    conn.commit()
    return read, inserted


def _enrich(conn, path: Path, enrich_cols: list[str], chunk_size: int,
            widths: dict[str, int | None] | None = None) -> int:
    """Fill genre/artist/popularity/year from the catalog, where it overlaps."""
    with conn.cursor() as cur:
        coldefs = ", ".join(
            f"{c} {'integer' if c in ('popularity', 'year') else 'text'}"
            for c in enrich_cols)
        cur.execute(f"""
            CREATE UNLOGGED TABLE IF NOT EXISTS _enrich_staging (
                track_id text PRIMARY KEY, {coldefs}
            );
            TRUNCATE _enrich_staging;
        """)
    conn.commit()

    cols = ["track_id"] + enrich_cols
    copy_sql = f"COPY _enrich_staging ({','.join(cols)}) FROM STDIN"
    sets = ", ".join(f"{c} = s.{c}" for c in enrich_cols)
    merge_sql = (f"UPDATE track_features t SET {sets} FROM _enrich_staging s "
                 f"WHERE t.track_id = s.track_id")

    updated = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        usable = [c for c in cols if c in (reader.fieldnames or [])]
        if "track_id" not in usable or len(usable) < 2:
            print(f"  {path.name} has none of the enrich columns; skipping")
            return 0

        batch = []
        for row in reader:
            batch.append(tuple(_coerce(c, row.get(c), (widths or {}).get(c))
                               for c in cols))
            if len(batch) >= chunk_size:
                updated += _flush(conn, batch, copy_sql, merge_sql, "_enrich_staging")
                batch.clear()
        if batch:
            updated += _flush(conn, batch, copy_sql, merge_sql, "_enrich_staging")

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _enrich_staging")
    conn.commit()
    return updated


# Counted rather than silently swallowed, so the seed report says how much
# real data got clipped to fit the schema.
TRUNCATED: dict[str, int] = {}


def _coerce(col: str, value, max_len: int | None = None):
    if value in (None, "", "nan"):
        return None
    if col in ("popularity", "year"):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    # Real catalogues carry tracks credited to dozens of artists, well past
    # the 300 characters models.py allows. Clipping is the right call for a
    # display field, but it is data loss and should be reported, not hidden.
    if max_len and len(value) > max_len:
        TRUNCATED[col] = TRUNCATED.get(col, 0) + 1
        return value[:max_len]
    return value


def _flush(conn, rows, copy_sql, merge_sql, staging) -> int:
    """COPY a chunk into staging, merge, commit. Interruptible between chunks."""
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as cp:
            for row in rows:
                cp.write_row(row)
        cur.execute(merge_sql)
        n = cur.rowcount
        cur.execute(f"TRUNCATE {staging}")
    conn.commit()
    return n


if __name__ == "__main__":
    sys.exit(main())
