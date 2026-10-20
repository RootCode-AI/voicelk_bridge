# Checkpoints go here

This folder is where the service looks for trained runs when nothing else is
configured. **Nothing inside it is committed** — checkpoints are hundreds of
megabytes each, far above what a git host accepts, and they change on their own
schedule anyway.

Each run is one folder holding its `config.json` and one checkpoint file. Folder
names are matched by prefix, so keep the training-run names:

```
models/
├── voicelk_vits_speaker_01/        config.json + best_model.pth
├── voicelk_vits_speaker_02/        config.json + best_model.pth
└── voicelk_vits_speaker_03/        config.json + best_model.pth (+ speakers.pth)
```

A checkpoint without the config from the same training run cannot be loaded
reliably, so always copy the two together.

## Getting them onto a server

Make an inference-only copy first — it drops the optimizer state, which serving never
reads, and cuts the file to a fraction of its size:

```bash
python tools/slim_checkpoint.py --model-dir "/path/to/voicelk_vits_speaker_03" \
    --out-dir "/path/to/deploy/voicelk_vits_speaker_03"
```

Then copy the slim run folder across:

```bash
rsync -avz --progress "/path/to/deploy/voicelk_vits_speaker_03/" \
    user@server:/srv/voicelk/models/voicelk_vits_speaker_03/
```

Point the service at that folder with `VOICELK_MODELS_ROOT`, or mount it at `/models`
when running in a container.

## Memory

Each loaded model occupies its own memory, so a server that serves every key needs
room for all of them. On a small box, register fewer models or leave the others unloaded
by never requesting them.
