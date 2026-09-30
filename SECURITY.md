# Security Policy

## Supported versions

Security fixes go to the latest commit on the `main` branch.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report privately via GitHub's
**Security → Report a vulnerability** (private advisory) on this repository instead. Include:

- what you found, and the file or line it affects;
- steps to reproduce it;
- the impact you expect.

You should receive an acknowledgement within 7 days. We will agree on a disclosure timeline with you
before anything is published.

## Security design

| Area | Measure |
|---|---|
| **Secrets** | The project needs **no API keys, tokens or credentials**. `.env` files are git-ignored, and `.env.example` contains only harmless defaults. A test (`tests/test_security.py`) scans every tracked file for key-like strings. |
| **Configuration** | `SwarmConfig` is a frozen dataclass. Every field is type-checked and range-checked when it is created, and invalid values raise `ValueError`. |
| **Environment settings** | `swarmrescue/settings.py` reads only the known keys (`SWARM_SEED`, `LOG_LEVEL`) and bounds or allow-lists every value. Bad values fall back to safe defaults. The `.env` file is parsed as text, never executed, and ignored if it is larger than 64 KB. |
| **CLI input** | Every numeric argument has an explicit range. Seed lists and PSO budgets are capped, which prevents resource exhaustion (for example, 10,000 seeds or 1,000 particles are rejected). Invalid input exits with code 2 and a clear message. |
| **Dashboard input** | All Streamlit controls are sliders, checkboxes or number inputs with fixed minimum and maximum values. There are no free-text fields. The PSO budget is capped, and the resulting config is validated again before anything runs. |
| **Web demo** | `web/index.html` is a single static file with inline CSS and JS. It makes no network requests and uses no cookies, storage, `eval` or third-party scripts. `web/vercel.json` sends a strict Content-Security-Policy and other security headers. |
| **Network** | The simulator, optimiser and advisor run fully offline. Streamlit usage statistics are disabled in `.streamlit/config.toml`. |
| **Dependencies** | All dependency versions are pinned in `requirements.txt`. |

## Out of scope

SwarmRescue is a research simulator. It does not control real robots, and its outputs are decision
support for trained operators, not safety-certified commands.
