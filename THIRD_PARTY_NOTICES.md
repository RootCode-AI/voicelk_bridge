# Third-party notices

The bridge ships the source of two third-party libraries under `vendor/`. They are
copied from the VoiceLK research repository by `tools/sync_from_ml.py` and keep their
own licenses; the MIT license in [`LICENSE`](LICENSE) does not apply to them.

| Path | Project | Version | License |
| --- | --- | --- | --- |
| `vendor/TTS/` | [Coqui TTS](https://github.com/coqui-ai/TTS) | 0.22.0 | [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) |
| `vendor/trainer/` | [Coqui Trainer](https://github.com/coqui-ai/Trainer) | 0.0.36 | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

`vendor/model_engine/` (the text normaliser, code-switched G2P and lexicon) is VoiceLK's
own code, generated from the research repository, and falls under the MIT license.

The MPL-2.0 is file-level copyleft: any change made to a file under `vendor/TTS/` must
be published under the MPL-2.0. Such changes belong in the research repository, never
in this copy — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

Trained checkpoints are not distributed with this repository. Their use is governed by
the terms of the datasets they were trained on.
