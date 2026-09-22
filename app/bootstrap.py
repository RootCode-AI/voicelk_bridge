"""
Makes the speech code importable from the bridge process.

Neither half of the ML stack is a pip package. The Coqui `TTS` library and the
`trainer` package are shipped as real source, and the text-processing modules import
each other by bare module name. Both folders therefore have to be on sys.path before
anything from them is imported.

By default the service uses its own vendored copies, which is what makes a checkout
runnable on a server without the research repository. Setting VOICELK_ML_ROOT points
it at a research checkout instead, so changes to the pipeline can be tried without
syncing first.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from .config import settings

LOGGER = logging.getLogger(__name__)

_prepared = False


def source_paths() -> List[Path]:
    """Folders that must be importable: the TTS/trainer packages, then the front-end."""
    return [settings.library_dir, settings.frontend_dir]


def ensure_ml_paths() -> None:
    """Idempotently place the speech source folders at the front of sys.path."""
    global _prepared
    if _prepared:
        return

    missing = [path for path in source_paths() if not path.is_dir()]
    if missing:
        origin = (
            "Set VOICELK_ML_ROOT to a research checkout, or unset it to use the bundled copies."
            if not settings.uses_bundled_sources
            else "Run the sync tool to populate the bundled sources from the research repository."
        )
        raise RuntimeError(
            "The speech sources were not found. "
            + origin
            + " Missing: "
            + ", ".join(str(path) for path in missing)
        )

    for path in source_paths():
        entry = str(path)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    _prepared = True
    LOGGER.debug(
        "Speech sources prepared (%s).",
        "bundled" if settings.uses_bundled_sources else "research checkout",
    )
