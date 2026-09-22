"""
Identity of the bundled text front-end.

The bridge carries its own copy of the normaliser, the G2P engine and the lexicon,
because a deployable service must not depend on a checkout of the research
repository. A copy can drift, and drift here is dangerous in a quiet way: if the
lexicon or a G2P rule changes after a model was trained, the model is fed phonemes
it never saw and simply mispronounces words, with nothing in the logs to say why.

So the copy is fingerprinted. The fingerprint is logged at startup and reported by
the health endpoint, and the sync tool records the fingerprint the research
repository had when the copy was taken. Comparing the two makes drift visible.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)

STAMP_FILENAME = "VENDOR_STAMP.json"

# Only the files that decide which phonemes a model receives take part.
FINGERPRINT_SUFFIXES = (".py", ".json")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digests(directory: Path) -> Dict[str, str]:
    """sha256 of every front-end source file, keyed by name, sorted for stability."""
    if not directory.is_dir():
        return {}

    digests: Dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in FINGERPRINT_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        digests[path.relative_to(directory).as_posix()] = file_digest(path)
    return digests


def fingerprint(directory: Path) -> str:
    """One short hash standing for the whole front-end."""
    digests = file_digests(directory)
    if not digests:
        return "unknown"

    combined = hashlib.sha256()
    for name, digest in digests.items():
        combined.update(name.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()[:16]


def read_stamp(vendor_root: Path) -> Optional[dict]:
    """The record the sync tool left when the copy was taken, if there is one."""
    stamp_path = vendor_root / STAMP_FILENAME
    if not stamp_path.is_file():
        return None
    try:
        return json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read the vendor stamp: %s", exc)
        return None


def drift_report(frontend_dir: Path, vendor_root: Path) -> dict:
    """
    Compares the live front-end against the stamp recorded at sync time.

    `status` is "clean" when they agree, "drifted" when the files changed since the
    copy was taken, and "unstamped" when there is nothing to compare against.
    """
    current = fingerprint(frontend_dir)
    stamp = read_stamp(vendor_root)

    if stamp is None:
        return {"status": "unstamped", "fingerprint": current}

    recorded = stamp.get("frontend_fingerprint")
    report = {
        "status": "clean" if recorded == current else "drifted",
        "fingerprint": current,
        "recorded_fingerprint": recorded,
        "synced_at": stamp.get("synced_at"),
        "source_commit": stamp.get("source_commit"),
    }
    return report
