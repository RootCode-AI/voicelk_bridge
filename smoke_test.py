"""
Manual smoke test for a running bridge.

Run it from the bridge folder while the service is up. It checks health, lists the
models, converts a sentence to phonemes and finally synthesizes a clip, writing the
result next to this script so you can listen to it.

    python smoke_test.py
    python smoke_test.py --model speaker_01 --text "ඔබට සුබ දවසක්" --out speaker_01.wav

Only the standard library is used, so it works from any interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote

DEFAULT_TEXT = "RAM එකක් නැතුව computer එකක් වැඩක් නෑ."


def call(url: str, api_key: str, payload: dict | None = None) -> tuple[int, bytes, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("X-API-Key", api_key)

    # Header names are case-insensitive and uvicorn sends them in lower case, so they
    # are keyed in lower case here rather than in the spelling the service sets.
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def show_json(title: str, status: int, body: bytes) -> None:
    print(f"\n=== {title} (HTTP {status}) ===")
    try:
        print(json.dumps(json.loads(body.decode("utf-8")), ensure_ascii=False, indent=2))
    except Exception:
        print(body[:500])


def main() -> int:
    # The output is Sinhala; a Windows console defaults to a code page that cannot show it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Smoke-test a running VoiceLK TTS Bridge.")
    parser.add_argument("--base-url", default=os.getenv("VOICELK_BRIDGE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VOICELK_BRIDGE_API_KEY", ""))
    parser.add_argument("--model", default=None, help="Model key; omit to use the service default.")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--out", default="smoke_test_output.wav")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    status, body, _ = call(f"{base}/health", args.api_key)
    show_json("health", status, body)
    if status != 200:
        print("The service is not reachable — start it first.")
        return 1

    status, body, _ = call(f"{base}/api/v1/models", args.api_key)
    show_json("models", status, body)

    status, body, _ = call(f"{base}/api/v1/phonemes", args.api_key, {"text": args.text})
    show_json("phonemes", status, body)

    payload = {"text": args.text, "speaker_id": args.speaker_id, "speed": args.speed}
    if args.model:
        payload["model"] = args.model

    print("\n=== synthesize (this may take a while the first time) ===")
    status, body, headers = call(f"{base}/api/v1/synthesize", args.api_key, payload)
    if status != 200:
        show_json("synthesize failed", status, body)
        return 1

    out_path = Path(args.out).resolve()
    out_path.write_bytes(body)

    print(f"HTTP {status}, {len(body)} bytes written to {out_path}")
    print(f"  model      : {headers.get('x-model-version')}")
    print(f"  duration   : {headers.get('x-audio-duration')} s")
    print(f"  render time: {headers.get('x-processing-time')} s")
    print(f"  normalized : {unquote(headers.get('x-normalized-text', ''))}")
    print(f"  phonemes   : {unquote(headers.get('x-ipa-sequence', ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
