"""Package metadata and small metadata helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PROJECT_NAME = "WaveSleuth-Devito"
__version__ = "0.9.3"
ARTIFACT_SCHEMA_VERSION = __version__


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def base_metadata() -> dict[str, Any]:
    """Return common metadata included in generated run/reconstruction files."""
    return {
        "project": PROJECT_NAME,
        "version": __version__,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": utc_timestamp(),
    }
