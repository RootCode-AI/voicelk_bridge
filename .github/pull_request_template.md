## What this changes

<!-- One or two sentences. Link the issue it resolves: "Closes #123". -->

## Why

<!-- The problem this solves, or the reason for the approach you chose. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Front-end sync from the research repository (`vendor/` only)
- [ ] Deployment / configuration
- [ ] Documentation

## How it was tested

<!-- Which model keys, speakers and inputs you tried. Include a code-switched sentence
     if you touched synthesis. -->

- [ ] `python smoke_test.py` passes against a real checkpoint and the clip sounds right
- [ ] `GET /health` reports the front-end as `clean`

## Checklist

- [ ] No hand edits under `vendor/` — or this PR is a sync, and names the research-repository commit it came from
- [ ] No checkpoints, `.env` files or secrets are committed
- [ ] `README.md` and `.env.example` are updated for any new endpoint, header or setting
- [ ] Any change to paths, response headers or error codes is called out below, because the backend depends on them

## API contract changes

<!-- "None", or what the backend needs to change. -->
