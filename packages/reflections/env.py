"""Canonical .env loading for Reflections."""

from __future__ import annotations

import os
from pathlib import Path

from reflections.config import REPO_ROOT


def load_env(path: Path | str | None = None) -> None:
    """Load key=value pairs from `.env` into ``os.environ``.

    **Semantics:** values from the file **override** any pre-existing shell
    environment variables. Treat `.env` as the single source of truth for
    local development. To force a shell override, remove the key from
    `.env` instead of relying on export order.
    """
    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value
    except FileNotFoundError:
        pass
