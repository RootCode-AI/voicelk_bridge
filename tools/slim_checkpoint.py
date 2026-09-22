"""
Strips a training checkpoint down to what inference actually needs.

A checkpoint saved during training carries the optimizer state, the learning-rate
schedule and the gradient scaler alongside the weights. The serving code never reads
any of that — it loads the weights and nothing else — yet the extra tensors are the
larger part of the file. Removing them makes a checkpoint far quicker to copy to a
server and lighter to keep around, without changing a single generated sample.

    python tools/slim_checkpoint.py --model-dir /path/to/run
    python tools/slim_checkpoint.py --model-dir /path/to/run --out-dir /path/to/deploy

The config file that belongs to the run is copied next to the slim checkpoint,
because a checkpoint without its own config cannot be loaded reliably.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE_DIR))

from app.bootstrap import ensure_ml_paths  # noqa: E402
from app.config import resolve_checkpoint  # noqa: E402

# Keys worth carrying over: the weights, plus the training position for provenance.
KEPT_KEYS = ("model", "step", "epoch")


def megabytes(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an inference-only copy of a checkpoint.")
    parser.add_argument("--model-dir", required=True, help="Training-run folder holding the config and checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename (default: the best one in the folder).")
    parser.add_argument("--out-dir", default=None, help="Where to write the slim run folder (default: <model-dir>-slim).")
    parser.add_argument("--name", default="best_model.pth", help="Filename for the slim checkpoint.")
    args = parser.parse_args()

    ensure_ml_paths()
    import torch

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"Model folder not found: {model_dir}")
        return 2

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print(f"No config file in {model_dir} — refusing to produce a checkpoint that cannot be loaded.")
        return 2

    try:
        source = resolve_checkpoint(model_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else model_dir.with_name(model_dir.name + "-slim")
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / args.name

    print(f"Reading  {source.name} ({megabytes(source):.0f} MB)")
    state = torch.load(source, map_location="cpu", weights_only=False)

    if not isinstance(state, dict) or "model" not in state:
        print("This file does not look like a training checkpoint (no weights entry found).")
        return 1

    dropped = sorted(key for key in state if key not in KEPT_KEYS)
    slim = {key: state[key] for key in KEPT_KEYS if key in state}

    torch.save(slim, destination)

    print(f"Dropped  {', '.join(dropped) if dropped else 'nothing'}")
    print(f"Wrote    {destination} ({megabytes(destination):.0f} MB)")

    config_copy = out_dir / "config.json"
    shutil.copy2(config_path, config_copy)
    print(f"Copied   {config_copy.name}")

    speakers = model_dir / "speakers.pth"
    if speakers.is_file():
        shutil.copy2(speakers, out_dir / speakers.name)
        print(f"Copied   {speakers.name}")

    print("\nThe slim folder is a drop-in replacement: point the model directory setting at it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
