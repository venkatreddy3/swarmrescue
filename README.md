# SwarmRescue

**Decentralized Ant-Pheromone Multi-Robot Exploration for Disaster Zone Search & Rescue**

After an earthquake, a swarm of robots enters a collapsed building and must cover every reachable
cell to find trapped survivors. There is no central controller: each robot decides its own next
move from what it senses and what it hears from nearby teammates. Particle Swarm Optimisation (PSO)
tunes the behaviour weights, and a rule-based Mission Advisor turns the results into plain-English
advice for the human operator.

---

## Chosen Vertical

**Track 05: Intelligent Systems & Autonomous Computing** (OptiForge 2026).

The project combines two computational-intelligence techniques:

- **Ant Colony-style stigmergy** for decentralized, self-organising exploration.
- **Particle Swarm Optimisation** for automatic parameter tuning.

---

## Problem Statement

A collapsed building is modelled as an occupancy grid (`0` = free, `1` = debris/wall). `N` robots
start at the entrance. They must:

1. **Cover** as much of the reachable free space as possible, as quickly as possible.
2. **Find** trapped survivors hidden in reachable cells.
3. **Never collide** with walls or with each other.
4. **Save energy.** Every move costs 1 battery unit, and a robot stops at 0.
5. **Survive change.** In **Round 2**, at tick 60 an aftershock drops 25 new debris cells.
   The robots are not told about them. Robot 0 fails permanently and the radio link goes down.

The mission score is:

```
fitness = 100*coverage + 20*survivors_found_ratio + 20*speed - 0.01*energy_moves - 100*collisions
speed   = 1 - tick_at_90pct_coverage / MAX_TICKS      (0 if 90% coverage is never reached)
```

`coverage` = visited reachable cells / reachable free cells. The reachable area is recomputed after
the Round 2 shift.

---

## Approach & Algorithmic Logic

### 1. Ant-pheromone move rule (per robot, every tick)

Each robot keeps a **private visit-count map** (its "pheromone"). It scores every free 4-neighbour
and moves to the one with the **lowest** score:

```
score(cell) = PHEROMONE_WEIGHT * visits[cell]
            + SPREAD_WEIGHT    * Σ_other 1 / (1 + manhattan(cell, other))   # crowding
            + RANDOMNESS       * U(0, 1)                                      # tie-break noise
```

- The **pheromone** term pushes robots away from cells that have already been searched. This is the
  ant-colony idea of *stigmergy* (coordinating through marks left in the environment), except that
  the marks are digital.
- The **crowding** term pushes robots apart so they fan out into different rooms.
- The **noise** term breaks symmetric ties so robots don't move in lock-step.

### 2. BFS fallback

If **all** neighbours have already been visited, the robot runs **Breadth-First Search** over its
own known-free cells to the **nearest unvisited** one and takes the first step. BFS on an unweighted
grid gives true shortest paths (checked by a test against an independent Bellman-Ford computation).
Perceived teammates are treated as obstacles, so the planner routes around them. When a robot's
belief contains no unvisited cells, it goes **idle** to save battery.

### 3. Safety and robustness

- **Collision-free by construction.** Robots move one at a time within a tick. A robot never enters
  a wall or a cell that is occupied or already claimed this tick, including cells held by failed
  robots. Because of this, two robots can never swap places either. The simulator also has an
  independent hardware interlock and audits every tick for shared cells, robots inside walls and
  head-on swaps.
- **Deadlock breaker.** If a robot has been blocked for 3 ticks, it takes a random free neighbour.
- **Sensing and replanning.** Every tick, each robot scans its surroundings within `SENSE_RANGE`
  (a square window). Debris that falls on a route the robot already knows is detected the next time
  the robot is nearby. The robot replans at once, because BFS runs again on the updated belief map.
- **Battery.** Each move costs 1, and a robot stops permanently at 0.

### 4. Decentralization

- Every robot owns its own `known` map (`-1` unknown / `0` free / `1` wall) and its own `visits` map.
  It never reads the ground-truth grid, only its sensor window.
- **Peer-to-peer sharing.** Robots within `COMM_RANGE` (Manhattan) merge both maps with `np.maximum`.
  This works because walls only ever appear: wall beats free, free beats unknown, and the higher visit
  count wins. Sharing is single-hop and uses a snapshot, so the result doesn't depend on the order
  robots are processed in.
- A robot only perceives teammates within `COMM_RANGE`. After the radio fails in Round 2, it only
  perceives teammates within 2 cells, using proximity sensors, and no maps are shared.

### 5. PSO tuning

PSO searches `PHEROMONE_WEIGHT ∈ [0,3]`, `SPREAD_WEIGHT ∈ [0,3]` and `RANDOMNESS ∈ [0,1]`:
8 particles, 12 iterations, inertia 0.6, c1 = c2 = 1.5. Velocity is clamped to 20% of each range.
The objective is the **fitness averaged over seeds 1, 2 and 3** (three different buildings), so the
weights aren't over-fitted to a single map. Particle 0 starts at the hand-set defaults, so the tuned
result can never be worse than the baseline on the tuning seeds. The global-best history is
monotone, which is also tested. A convergence line is printed every iteration.

### Why these CI techniques?

| Need | Technique | Why it fits |
|---|---|---|
| Explore without a leader, survive robot or radio loss | Ant-colony stigmergy | Purely local rules produce global coverage, and losing one robot or the radio link degrades performance gradually instead of breaking the mission |
| Fill gaps when local marks run out | BFS | Optimal (shortest) on unweighted grids, and fast enough to rerun every tick for instant replanning |
| Tune 3 continuous weights with a noisy, non-differentiable, simulation-based objective | PSO | Gradient-free, needs no model of the objective, works with few evaluations, and is itself a swarm-intelligence method |
| Explainable operator guidance with no network or API keys | Rule-based expert system | Deterministic, auditable, works offline in the field |

---

## How It Works End-to-End

```
           ┌─────────────────────── config.py (frozen SwarmConfig + validation) ─────────────────────┐
           ▼                                                                                          │
   world.py: random collapsed building ──► starts at entrance, survivors in reachable cells           │
           │                                                                                          │
           ▼          ┌──────────────────────── one tick (simulation.py) ────────────────────────┐   │
   ┌──────────────┐   │ [Round 2 @ tick 60: +25 debris, robot 0 fails, radio off, recompute reach]│   │
   │ Agent 0..N-1 │──►│ 1. sense walls in SENSE_RANGE        (agent.py)                           │   │
   │ known map    │   │ 2. share maps with peers in range    (coordination.py, np.maximum)       │   │
   │ visits map   │   │ 3. each robot in turn: perceive neighbours → choose_move                  │   │
   │ battery      │   │      pheromone score ─► BFS fallback ─► deadlock breaker ─► idle          │   │
   └──────────────┘   │ 4. safety interlock, move, battery−1, collision audit, coverage, survivors│   │
                      └───────────────────────────────┬───────────────────────────────────────────┘   │
                                                      ▼                                               │
                              SimulationResult (coverage, t@90%, energy, collisions, latency, fitness)│
                                          │                          │                                │
                                          ▼                          ▼                                │
                             optimizer.py: PSO over seeds 1,2,3 ─────┴──► tuned weights ──────────────┘
                                          │
                                          ▼
                             advisor.py: rules → plain-English recommendations
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                    ▼
                 main.py (CLI report)             app.py (Streamlit Mission Control)
```

---

## How to Run

Requires Python 3.10+ (developed on 3.13).

```bash
pip install -r requirements.txt
```

**CLI**

```bash
python main.py --round 1                 # static building, seeds 1 2 3
python main.py --round 2                 # aftershock + robot failure + radio loss
python main.py --round 1 --tune          # PSO tuning + before/after + held-out check
python main.py --round 2 --tune --seeds 1 2 3 --eval-seeds 101 102 103
python main.py --agents 6 --battery 150 --wall-density 0.25 --priority speed
```

**Tests** (90 tests, about 20 s)

```bash
python -m pytest
```

**Dashboard**

```bash
streamlit run app.py
```

In the sidebar, the operator sets the scenario (Round 1 or 2), number of robots, building size,
debris density, battery, survivors, time limit, seed and priority. The **Behaviour weights** section
also has manual weight sliders. The main view shows the map (tick slider and *Play animation*),
metrics, a coverage-over-time chart, a PSO convergence chart with an *Apply tuned weights* button,
and the Mission Advisor.

Accessibility: the map uses the colour-blind-safe **Okabe-Ito** palette, and every element also has
a distinct **shape**:

| Element | Shape | Colour |
|---|---|---|
| Robot (with its id) | circle | orange |
| Failed robot | X | vermillion |
| Survivor found | star | green |
| Survivor not yet found | triangle | purple |
| New Round 2 debris | grey cell with an "x" | grey |

A text legend explains all of these. Under the map, a plain-text summary lists the tick, coverage,
survivors found and every robot's position, for screen readers. Every control has a descriptive label.

---

## Results

These are real outputs of `python main.py --round N --tune` with the default configuration
(Windows 11, Python 3.13). Everything is seeded, so all metrics except wall-clock latency are
reproducible.

### Round 1: static building

| Seed | Coverage | Survivors | Tick @ 90% | Energy | Collisions | Fitness (default → tuned) |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 5/5 | 118 → 110 | 582 → 562 | 0 | 126.313 → 127.047 |
| 2 | 1.000 | 5/5 | 95 → 91 | 619 → 559 | 0 | 127.477 → 128.343 |
| 3 | 1.000 | 5/5 | 101 → 95 | 578 → 491 | 0 | 127.487 → 128.757 |
| **Mean** | **1.000** | **100%** | **105 → 99** | **593 → 537** | **0** | **127.092 → 128.049** |

Tuned weights: `pheromone_weight=2.173, spread_weight=0.579, randomness=0.636`.
PSO took 23.8 s. Max decision latency was ≤ 2.3 ms per tick for the whole swarm.

```
[PSO] iter  0 | best  127.532 | swarm mean  125.545 | pheromone_weight=2.572, spread_weight=0.101, randomness=0.730
[PSO] iter  2 | best  127.751 | swarm mean  127.258 | pheromone_weight=2.671, spread_weight=1.248, randomness=0.403
[PSO] iter  3 | best  128.049 | swarm mean  127.207 | pheromone_weight=2.173, spread_weight=0.579, randomness=0.636
...
[PSO] iter 12 | best  128.049 | swarm mean  127.595 | pheromone_weight=2.173, spread_weight=0.579, randomness=0.636
```

Held-out seeds 101–110 (10 maps never seen by PSO): fitness **127.289 → 127.331**. Coverage and
survivors stayed at 100%, energy dropped 580.4 → 560.9, and ticks to 90% went 103.6 → 105.9.
Round 1 is already near its ceiling, so tuning mainly saves energy.

### Round 2: aftershock + robot 0 failure + radio loss at tick 60

| Seed | Coverage | Survivors | Tick @ 90% | Energy | Collisions | Fitness (default → tuned) |
|---|---|---|---|---|---|---|
| 1 | 0.993 → 0.997 | 5/5 | 127 → 135 | 806 → 806 | 0 | 122.804 → 122.611 |
| 2 | 0.970 → 0.959 | 5/5 | 118 → 110 | 808 → 808 | 0 | 121.033 → 120.519 |
| 3 | 0.974 → 1.000 | 5/5 | 183 → 125 | 808 → 655 | 0 | 117.106 → 125.117 |
| **Mean** | **0.979 → 0.985** | **100%** | **143 → 123** | **807 → 756** | **0** | **120.314 → 122.749** |

Tuned weights: `pheromone_weight=0.522, spread_weight=0.984, randomness=0.394`.
PSO took 21.9 s. Max decision latency was ≤ 12.4 ms (one Windows scheduling spike; typically < 1 ms).

```
[PSO] iter  0 | best  121.362 | swarm mean  112.671 | pheromone_weight=2.805, spread_weight=2.448, randomness=0.003
[PSO] iter  1 | best  121.502 | swarm mean  118.495 | pheromone_weight=0.973, spread_weight=2.555, randomness=0.447
[PSO] iter  3 | best  122.410 | swarm mean  119.792 | pheromone_weight=0.902, spread_weight=1.171, randomness=0.478
[PSO] iter  4 | best  122.749 | swarm mean  117.396 | pheromone_weight=0.522, spread_weight=0.984, randomness=0.394
...
[PSO] iter 12 | best  122.749 | swarm mean  117.064 | pheromone_weight=0.522, spread_weight=0.984, randomness=0.394
```

Held-out seeds 101–110:

| Metric | Default | Tuned |
|---|---|---|
| Coverage | 0.973 | 0.982 |
| Survivors found | 94% | 100% |
| Ticks to 90% | 176.7 | 151.3 |
| Energy | 779.3 | 783.2 |
| Collisions | 0 | 0 |
| **Fitness** | **116.498** | **120.318** |

On unseen buildings, tuning found every survivor and reached 90% coverage about 25 ticks sooner.

Mission Advisor output of `python main.py --round 2` (default weights):

```
[INFO] Robots are revisiting cells: About 2.7 moves per explored cell. Without radio, robots cannot share pheromone maps and re-search each other's areas; restoring map sharing helps most.
[INFO] Plan for failures: Round 2 lost a robot and the radio link. Keep a spare robot in reserve and consider dropping radio relays so robots can keep sharing maps.
[SUCCESS] Mission on track: 98% coverage, 100% of survivors found, zero collisions.
```

With `--tune`, the advisor reports on the tuned weights instead. That run shows 2.6 moves per explored
cell and 99% coverage, and adds: `[INFO] Apply tuned weights: PSO improved mean fitness from 120.31 to 122.75.`

**Takeaways**

- There were **zero collisions** in every run. The tests also check 25 seeds × 2 rounds on a small
  map plus 8 seeds × 2 rounds on the default map.
- Round 2 costs about 7 fitness points: debris, one lost robot and no map sharing.
- In Round 2, PSO learned a *lower* pheromone weight and a *higher* spread weight. Without radio,
  robots can't see each other's pheromone, so keeping physically apart matters more than
  a robot's own visit history.

---

## Parameter Configuration

All parameters live in the frozen dataclass `SwarmConfig` (`swarmrescue/config.py`). Out-of-range or
wrongly typed values raise `ValueError`.

| Parameter | Default | Valid range | Meaning |
|---|---|---|---|
| `grid_size` | 20 | 5–100 | Side length of the square grid |
| `wall_density` | 0.18 | 0–0.45 | Initial debris fraction |
| `num_agents` | 4 | 1–20 | Robots |
| `num_survivors` | 5 | 0–50 | Trapped survivors |
| `max_ticks` | 300 | 1–5000 | Mission time limit |
| `battery` | 250 | ≥ 1 | Moves per robot |
| `comm_range` | 5 | 1–2·grid | Manhattan radio range (map sharing, teammate perception) |
| `sense_range` | 1 | 1–5 | Chebyshev wall-sensor radius |
| `pheromone_weight` | 1.0 | 0–3 | Visit-count penalty (PSO-tuned) |
| `spread_weight` | 0.5 | 0–3 | Crowding penalty (PSO-tuned) |
| `randomness` | 0.1 | 0–1 | Noise amplitude (PSO-tuned) |
| `shift_tick` | 60 | ≥ 0 | Round 2 shift tick |
| `new_walls` | 25 | 0–grid²/4 | Debris cells added at the shift |
| `seed` | 0 | ≥ 0 | Master seed (map, survivors, noise, shift) |

The feasibility check also requires `num_agents + num_survivors ≤ free_area / 4`.

---

## Assumptions & Operational Constraints

- **Grid world.** Robots use 4-connected moves, one cell per tick, and every move costs 1 battery unit.
- **Entrance.** Robots start at the reachable cells closest to the top-left corner. Maps whose
  reachable area is below 40% of the free space are regenerated.
- **Survivors.** A survivor counts as found when a robot enters their cell. The survivor ratio is
  measured against all survivors, including any that debris cuts off in Round 2.
- **Sensing.** Wall sensing is perfect within the square window. Robots never see the full map.
- **Round 2 coverage.** After the shift, the reachable area is the free cells connected to a working
  robot (a failed robot counts as an obstacle), plus cells already visited that are still free.
- **Communication.** Radio is single-hop within `comm_range` and has no latency or packet loss while
  it works. After it fails, a robot only perceives teammates within 2 cells and no maps are merged.
- **Failed robots** stay in place as permanent obstacles and never move, share or sense again.
- **Simulator-only.** The "physical interlock" in the simulator never fired in any test. It is there
  to mirror the bumper or safety stop a real robot would have.
- **Latency** is wall-clock time for the swarm's sense, share and decide cycle in one tick. It depends
  on the machine, but stays far below the 50 ms budget (and a test enforces that budget).
- **Determinism.** A given `(config, seed)` always produces the same mission. The map, agent noise
  and shift use independent `SeedSequence` streams, so changing the weights never changes the building.
- **Privacy and security.** Everything runs locally. There are no network calls, API keys or secrets,
  and Streamlit usage statistics are turned off.

---

## Attempt Notes

What changed between iterations, and why:

1. **Attempt 1: baseline swarm.** Implemented the pheromone score, BFS fallback, deadlock breaker,
   sequential collision-free moves and `np.maximum` map merging. The first run on the defaults gave
   100% coverage and 0 collisions in Round 1.
2. **Idle mode instead of endless wandering.** Once a robot's belief contains no unvisited cells,
   it stays put instead of drifting to the least-visited neighbour. This cuts wasted moves, which
   carry a 0.01 fitness penalty each. It also lets the simulator stop early once every robot is idle
   or everything is covered, which made the 312-simulation PSO run take about 25 s.
3. **BFS routes around teammates.** Perceived robots are obstacles for BFS, so a robot takes a
   detour instead of waiting behind a teammate. The 3-tick deadlock breaker is kept as a last resort.
4. **Fairer Round 2 coverage.** Recomputing reachability only from the robots' positions would
   ignore work that was already done. The denominator now also keeps cells that were visited before
   the aftershock, and it treats the failed robot as an obstacle.
5. **PSO seeded with the defaults and checked on held-out maps.** Putting the hand-set weights in
   particle 0 guarantees tuning never makes things worse on the tuning seeds. The CLI then evaluates
   on 10 unseen seeds to check for over-fitting. This showed the Round 1 gain is marginal
   (+0.04 fitness) and the Round 2 gain is real (+3.8).
6. **Advisor contradicted PSO.** In Round 2 the "revisiting cells" rule said to *raise* the pheromone
   weight while PSO had just *lowered* it. The real cause is the lost radio link, so that rule is now
   round-aware and recommends restoring map sharing instead.
7. **Dashboard readability.** In the first layout, the metric labels were cut off in four narrow
   columns, so it now uses a two-column grid. The map also got integer row/column axes and a hatched
   legend entry for new debris.
8. **Test fix.** One fitness assertion used a hand-computed 92.0. The correct value is
   80 + 12 + 10 − 5 = 97.0, so the test was wrong, not the code.

---

## Project Structure

```
swarmrescue/
├── swarmrescue/
│   ├── __init__.py
│   ├── config.py        # SwarmConfig frozen dataclass + validation, PSO bounds
│   ├── world.py         # map generation, BFS distances, reachable(), apply_shift()
│   ├── agent.py         # Agent: sensing, pheromone scoring, choose_move, BFS, deadlock breaker
│   ├── coordination.py  # decentralized perception + peer-to-peer map sharing
│   ├── simulation.py    # simulate() -> SimulationResult, fitness, coverage, collision audit
│   ├── optimizer.py     # PSO with convergence log
│   └── advisor.py       # rule-based Mission Advisor
├── tests/
│   ├── conftest.py      # shared fixtures
│   ├── test_config.py   # defaults, immutability, validation
│   ├── test_world.py    # generation, BFS distances, reachability, shift
│   ├── test_agent.py    # BFS optimality, pheromone/spread rules, deadlock, replanning, sharing
│   ├── test_simulation.py # fitness formula, zero collisions, walls, dead robot, latency, determinism
│   ├── test_optimizer.py  # monotone best, bounds, reproducibility
│   ├── test_advisor.py    # recommendation rules
│   ├── test_cli.py        # CLI smoke tests
│   └── test_app.py        # headless Streamlit AppTest
├── main.py              # CLI
├── app.py               # Streamlit "Rescue Mission Control"
├── requirements.txt     # pinned dependencies
├── pytest.ini
├── .streamlit/config.toml
├── .gitignore
├── LICENSE
└── README.md
```
