# Changelog

## Attempt 3 (in progress)
- **Streamlit Cloud crash fixed for good.** The server kept already-imported Attempt 1/2 `swarmrescue` modules alive across `git pull`s. `app.py` now puts the repo root first on `sys.path` and purges stale modules before importing, and it versions and clears its caches. `requirements.txt` never installs the project, and a test enforces that.
- **Code quality.** Added `pyproject.toml` with ruff (lint and format) and `mypy --strict`, both clean. The advisor is now a rule engine, robot decisions are an ordered behaviour chain, and `app.py` is split into `swarmrescue/render.py` and `swarmrescue/ui_state.py`. A test enforces functions of at most 40 code lines, docstrings and type hints.
- **Testing.** Added Hypothesis property tests and a coverage gate (96.8% coverage). GitHub Actions CI runs ruff, mypy, pytest with coverage, bandit and pip-audit.
- **Efficiency.** Perception and pheromone scoring are vectorised, and a validated BFS route cache runs 31% fewer BFS searches. `scripts/benchmark.py` shows a worst tick of 17 ms, against a 50 ms budget.
- **Problem alignment.** Added `safety_margin`, deadlock counters, latency-budget checks, trajectory-recalibration time, throughput and coverage velocity. `docs/REQUIREMENTS_TRACEABILITY.md` is checked by a test.
- **Not included:** the energy-aware feature (a low-battery relay or frontier yielding) was prototyped but not shipped.

## Attempt 2 (score 67.17)
- Added an instant-load Vercel demo (`web/index.html`): a live JavaScript port of the swarm with an Aftershock button and a strict CSP.
- Renamed the code to domain terms: `DisasterZone`, `Robot`, `PheromoneTrail`, `RadioLink`, `Aftershock`, `MissionControl`, `MissionAdvisor`.
- New features: pheromone evaporation (off by default) and survivor acoustic pings (on by default), plus a time-to-first-survivor metric and an ablation script.
- Added a PSO over-fitting guard that checks tuned weights on held-out maps.
- Security: a safe `.env` settings loader, `.env.example`, `SECURITY.md`, bounded CLI and dashboard inputs, and a secret-scan test.
- Docs: SDG 9 and 11 alignment, Mermaid diagrams and screenshots.
- Fixed the Streamlit `TypeError` from stale keys, and the end-of-mission message now states the real reason the mission ended.

## Attempt 1
- Built the decentralized swarm: pheromone move rule, BFS fallback, 3-tick deadlock breaker and collision-free sequential moves.
- Robots merge maps within radio range using `np.maximum`.
- Added the Round 2 aftershock: new debris, robot 0 fails and the radio drops.
- Implemented the fitness function, latency measurement, PSO tuning, a rule-based advisor, the CLI and the Streamlit dashboard.
