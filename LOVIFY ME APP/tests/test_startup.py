"""The app must refuse to start without a SECRET_KEY.

Carried forward from the security fix on the SECRET_KEY branch. The failure
mode it prevents is not a crash but the opposite: silently falling back to a
literal key that is committed in this repository's public history, which
would let anyone forge a session cookie -- and now also an OAuth state,
since both are signed with this key.

Run in a subprocess because the check happens at import, and `lovify` is
already imported by the rest of the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


# A developer tree has a real .env at the repository root, and load_dotenv()
# walks up to find it -- so an empty `env=` is not an empty environment. The
# stub neutralises that one call; everything else about the import is real.
IMPORT_WITHOUT_DOTENV = (
    "import sys, types;"
    "m = types.ModuleType('dotenv'); m.load_dotenv = lambda *a, **k: None;"
    "sys.modules['dotenv'] = m;"
    "import lovify"
)


def _import_lovify_with(env_overrides: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
    env.update(PYTHONPATH=str(APP_DIR), **env_overrides)
    return subprocess.run([sys.executable, "-c", IMPORT_WITHOUT_DOTENV],
                          capture_output=True, text=True, env=env, cwd=APP_DIR)


def test_no_secret_key_is_a_refusal_to_start_not_a_fallback():
    done = _import_lovify_with({})

    assert done.returncode != 0
    assert "SECRET_KEY is not set" in done.stderr
    # The message has to tell someone what to do about it.
    assert "secrets.token_hex" in done.stderr


def test_with_a_secret_key_the_app_imports_cleanly(tmp_path):
    done = _import_lovify_with({
        "SECRET_KEY": "a-key",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'boot.db'}",
    })

    assert done.returncode == 0, done.stderr
