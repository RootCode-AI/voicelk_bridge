# Security Policy

## Supported versions

This is a research project with a single maintained line. Security fixes go into
`main` only.

| Version | Supported |
| --- | --- |
| `main` (1.x) | Yes |
| Anything older | No |

## Reporting a vulnerability

**Please do not open a public issue, discussion or pull request for a security
problem.**

Report it privately through GitHub:
**[Report a vulnerability](https://github.com/Tharinda-Pamindu/bridge/security/advisories/new)**
(the *Security* tab → *Report a vulnerability*).

Please include:

- What the problem is and what an attacker could do with it.
- Steps or a request that reproduces it, and the commit you tested.
- How the service was deployed (Docker, systemd, Windows script) and which settings
  were relevant — never paste a real API key.

You can expect an acknowledgement within 7 days and an assessment within 14. Once a
fix is ready, the advisory is published with credit to you, unless you prefer not
to be named.

## Scope

In scope — the code in this repository:

- Bypassing the `X-API-Key` check.
- Requests that crash the process, exhaust memory or hold a model indefinitely
  despite `VOICELK_MAX_TEXT_CHARS`.
- Path traversal or arbitrary file reads through model keys or configuration.
- Header injection through the percent-encoded response headers.
- Unsafe defaults in `Dockerfile`, `docker-compose.yml` or `deploy/`.

Out of scope:

- A deployment run without `VOICELK_BRIDGE_API_KEY`, or exposed on a public network.
  The bridge is an internal service; the documentation requires a key and a loopback
  or private-network binding.
- `VOICELK_ALLOWED_ORIGINS=*` on a development machine.
- Vulnerabilities in the vendored Coqui `TTS` or `trainer` code that are not reachable
  through the bridge's API. Report those upstream.
- Loading a malicious checkpoint. Checkpoints are PyTorch pickles and are trusted
  input; only load weights you produced or obtained from a trusted source.

## Deployment guidance

- Always set `VOICELK_BRIDGE_API_KEY` to a long random value and send it only from the
  backend.
- Publish the port on loopback or a private network — the compose file already binds
  to `127.0.0.1`.
- Mount checkpoints read-only, as the compose file does.
- Keep `.env` out of git; only `.env.example` is committed.
