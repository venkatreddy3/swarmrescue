# Changelog

## Attempt 2 (after Attempt 1 scored 67.17)
- **Fast demo.** Added an instant-load Vercel page (`web/index.html`): a live JavaScript port of the swarm with an Aftershock button and a strict CSP.
- **Domain vocabulary.** Renamed the code to domain terms: `DisasterZone`, `Robot`, `PheromoneTrail`, `RadioLink`, `Aftershock`, `MissionControl`, `MissionAdvisor`.
- **Innovation.** Pheromone evaporation (off by default) and survivor acoustic pings (on by default), a time-to-first-survivor metric, and an ablation script. An energy-aware behaviour was prototyped but not shipped.
- **PSO over-fitting guard.** Tuned weights must not lose fitness on held-out maps.
- **Security.** A safe `.env` settings loader, `.env.example` and `SECURITY.md`. CLI and dashboard inputs are bounded, dependencies are pinned exactly, a test scans for secrets, and bandit and pip-audit are clean.
- **Streamlit Cloud crash fixed for good.** The server kept stale, already-imported `swarmrescue` modules across `git pull`s. `app.py` now purges them before importing, and versions and clears its caches. The dashboard passes only accepted config fields, and `requirements.txt` never installs the project, which a test enforces.
- **Code quality.** `pyproject.toml` configures ruff and `mypy --strict`, both clean. The advisor is a rule engine, robot decisions are a behaviour chain, and `app.py` is split into `render.py` and `ui_state.py`. Functions have at most 40 code lines, which a test enforces.
- **Testing.** Added Hypothesis property tests (96.8% coverage) and GitHub Actions CI running ruff, mypy, pytest with coverage, bandit and pip-audit.
- **Efficiency.** Perception and scoring are vectorised, and a validated BFS route cache runs 31% fewer searches. `scripts/benchmark.py` shows a worst tick of 17 ms, against a 50 ms budget.
- **Problem alignment.** Added `safety_margin`, deadlock counters, latency-budget checks, recalibration time, throughput and coverage velocity, plus `docs/REQUIREMENTS_TRACEABILITY.md`, which a test checks.
- **Messages and docs.** End-of-mission messages give the real reason. Added the SDG 9 and 11 section, Mermaid diagrams, screenshots, live demo links, and dashboard help text.

## Attempt 1 (score 67.17)
- Built the decentralized swarm: pheromone move rule, BFS fallback, 3-tick deadlock breaker and collision-free sequential moves.
- Robots merge maps within radio range using `np.maximum`.
- Added the Round 2 aftershock: new debris, robot 0 fails and the radio drops.
- Implemented the fitness function, latency measurement, PSO tuning, a rule-based advisor, the CLI and the Streamlit dashboard.
