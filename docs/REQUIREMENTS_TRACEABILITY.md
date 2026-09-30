# Requirements Traceability Matrix

Every line of the **Track 05: Intelligent Systems & Autonomous Computing** specification and the SwarmRescue
project brief is mapped to the module and function that implement it and to the test that proves it.

`tests/test_traceability.py` parses this file and fails if any referenced file, function, class or test no
longer exists, so the matrix cannot drift away from the code. The evidence column gives measured numbers
from the README (`Results`, `Performance`, `Innovation`).

## 1. Hard constraints

| Spec line | Implementation (module :: function) | Verified by (test) | Evidence |
|---|---|---|---|
| Collision-free trajectories | `swarmrescue/simulation.py::MissionControl._move_robots` (sequential moves and an occupancy check) and `swarmrescue/agent.py::Robot.choose_move` (never picks a perceived robot's cell) | `tests/test_properties.py::test_any_mission_is_collision_free_and_bounded`, `tests/test_simulation.py::test_zero_collisions_and_valid_coverage_many_seeds`, `tests/test_agent.py::test_never_moves_into_debris_or_robots` | 0 collisions in every run, including all Hypothesis-generated maps |
| Collision audit (no shared cells, no robots in debris, no swaps) | `swarmrescue/simulation.py::count_collisions` | `tests/test_simulation.py::test_robots_never_occupy_walls_or_share_cells` | Interlock trips are always 0 |
| Safe operational margins | `swarmrescue/config.py::SwarmConfig` (`safety_margin`), `swarmrescue/agent.py::Robot._keeps_clearance`, `swarmrescue/agent.py::Robot._enforce_margin`, `swarmrescue/telemetry.py::min_pairwise_distance` | `tests/test_alignment.py::test_clearance_rule`, `tests/test_alignment.py::test_safety_margin_reduces_close_contact_without_collisions`, `tests/test_alignment.py::test_margin_blocks_planned_step_but_deadlock_breaker_may_relax_it` | `safety_margin=1` cuts adjacent-robot ticks from 6.2% to 1.1% (Round 1) |
| Decision latency within the real-time budget | `swarmrescue/config.py::SwarmConfig` (`latency_budget_ms`), `swarmrescue/telemetry.py::MissionTelemetry.record_tick` (counts and logs violations), `swarmrescue/advisor.py::rule_latency` | `tests/test_alignment.py::test_latency_budget_warning`, `tests/test_alignment.py::test_default_missions_meet_the_real_time_budget`, `tests/test_simulation.py::test_latency_per_tick_under_50ms`, `tests/test_efficiency.py::test_benchmark_latency_within_real_time_budget` | Worst tick 17.3 ms (16 robots, 40×40), against a 50 ms budget |
| No centralized single point of failure | `swarmrescue/agent.py::Robot.choose_move` (local decisions only), `swarmrescue/coordination.py::RadioLink.share_pheromone_trails` (peer-to-peer), `swarmrescue/simulation.py::MissionControl.trigger_aftershock` (failure injection) | `tests/test_simulation.py::test_round2_shift_effects`, `tests/test_simulation.py::test_mission_control_records_aftershock`, `tests/test_agent.py::test_failed_robot_does_not_transmit` | Round 2 loses robot 0 and the radio link and still finds 99% of survivors |
| Sub-second trajectory recalibration | `swarmrescue/agent.py::Robot.sense_debris` (detects debris on a known route), `swarmrescue/agent.py::Robot._reroute` (BFS replanning in the same tick), `swarmrescue/telemetry.py::MissionTelemetry.record_decision` (measures it) | `tests/test_alignment.py::test_trajectory_recalibration_is_sub_second`, `tests/test_agent.py::test_reroute_after_new_debris`, `tests/test_efficiency.py::test_new_debris_on_route_forces_a_new_bfs` | Maximum reroute time 0.54 ms (about 1/2000 of a second) |

## 2. Objectives

| Spec line | Implementation (module :: function) | Verified by (test) | Evidence |
|---|---|---|---|
| Throughput | `swarmrescue/simulation.py::SimulationResult.throughput` | `tests/test_alignment.py::test_throughput_and_coverage_velocity` | 2.27 cells/tick (Round 1, 20 maps) |
| Coverage velocity | `swarmrescue/simulation.py::SimulationResult.coverage_velocity`, `swarmrescue/simulation.py::compute_fitness` (speed term) | `tests/test_alignment.py::test_throughput_and_coverage_velocity`, `tests/test_simulation.py::test_fitness_formula_exact` | 0.69 coverage points/tick (Round 1) |
| Collision risk | `swarmrescue/simulation.py::count_collisions`, `swarmrescue/telemetry.py::MissionTelemetry.record_positions` (minimum separation) | `tests/test_properties.py::test_any_mission_is_collision_free_and_bounded` | 0 collisions; minimum separation reported every run |
| Deadlocks | `swarmrescue/agent.py::Robot._break_deadlock` (3-tick breaker), `swarmrescue/telemetry.py::MissionTelemetry.record_decision` (counters) | `tests/test_agent.py::test_deadlock_breaker_after_three_blocked_ticks`, `tests/test_alignment.py::test_telemetry_counts_deadlocks_and_blocked_ticks` | About 2.4 deadlocks detected and broken per Round 2 mission, with the swarm never freezing |
| Energy | `swarmrescue/agent.py::Robot.commit` (1 battery unit per move), `swarmrescue/simulation.py::compute_fitness` (−0.01 per move) | `tests/test_simulation.py::test_battery_limits_energy`, `tests/test_agent.py::test_failed_or_empty_robot_never_moves` | Energy is reported for every mission |
| Path length | `swarmrescue/simulation.py::SimulationResult` (`mean_path_length`), `swarmrescue/agent.py::bfs_path` (shortest routes) | `tests/test_alignment.py::test_mean_path_length_is_energy_per_robot`, `tests/test_properties.py::test_bfs_path_length_equals_true_shortest_distance` | BFS matches an independent Bellman-Ford on every generated grid |
| Resilience to agent failure | `swarmrescue/agent.py::Robot.fail`, `swarmrescue/coordination.py::perceive_teammates` (failed robot = obstacle), `swarmrescue/simulation.py::post_aftershock_reachable` | `tests/test_agent.py::test_perception_includes_failed_robots_as_obstacles`, `tests/test_simulation.py::test_replanning_after_shift_keeps_exploring` | Round 2 coverage 98% after losing robot 0 |
| Resilience to communication dropout | `swarmrescue/coordination.py::RadioLink.cut`, `swarmrescue/coordination.py::RadioLink.perception_radius` (falls back to 2-cell proximity sensing) | `tests/test_agent.py::test_radio_link_cut_stops_sharing` | Round 2 runs its whole second half without radio |
| Online learning (agents that learn) | `swarmrescue/learning.py::AdaptiveWeightLearner`, `swarmrescue/simulation.py::MissionControl._attach_learners` (one learner per robot) | `tests/test_learning.py::test_greedy_learner_tries_every_preset_then_exploits_the_best`, `tests/test_learning.py::test_each_robot_learns_independently` | Round 1 fitness 126.997 → 127.321, Round 2 117.093 → 117.401, energy −5% (seeds 1–3) |

## 3. Evaluation

| Spec line | Implementation | Verified by (test) | Evidence |
|---|---|---|---|
| Multi-scenario benchmarks | `scripts/benchmark.py::measure` (4/8/16 robots × 20×20/40×40 × Round 1/2), `scripts/ablation.py::main` (feature variants on 20 maps), `main.py::held_out_check` (10 unseen maps) | `tests/test_efficiency.py::test_benchmark_latency_within_real_time_budget`, `tests/test_simulation.py::test_ablation_script_runs` | README tables: Performance, Innovation, Results |
| Runtime | `swarmrescue/telemetry.py::MissionTelemetry.record_tick`, `scripts/benchmark.py::LatencyRow` (mean/p95/max) | `tests/test_simulation.py::test_latency_per_tick_under_50ms` | Mean 0.3–2.7 ms per tick |
| Stability under perturbation | `scripts/benchmark.py::stability_row` (six perturbations, 20 seeds each, fitness mean ± std) | `tests/test_properties.py::test_any_mission_is_collision_free_and_bounded` (random perturbations) | Round 1 std 0.8; collisions 0 under every perturbation |

## 4. Submission checklist

| Checklist item | Where | Verified by (test) |
|---|---|---|
| Parameter configuration | `swarmrescue/config.py::SwarmConfig` (frozen, validated), `swarmrescue/config.py::validate_config`, README "Parameter Configuration" table | `tests/test_config.py::test_defaults_match_specification`, `tests/test_config.py::test_validation_rejects_bad_values`, `tests/test_properties.py::test_config_rejects_every_invalid_weight` |
| Fitness per attempt | `swarmrescue/simulation.py::compute_fitness`; README "Results" (Attempt 1 and 2); `CHANGELOG.md` | `tests/test_simulation.py::test_fitness_formula_exact`, `tests/test_simulation.py::test_result_fitness_is_consistent` |
| Convergence evidence | `swarmrescue/optimizer.py::IterationLog.format` (logs every iteration), `swarmrescue/optimizer.py::run_pso`, dashboard convergence chart (`app.py::show_pso`) | `tests/test_optimizer.py::test_best_fitness_never_decreases`, `tests/test_optimizer.py::test_convergence_log_emitted`, `tests/test_cli.py::test_cli_tune_prints_convergence_log` |
| Representation rationale | README "Approach & Algorithmic Logic" (occupancy grid, per-robot belief map, pheromone trail), `docs/ARCHITECTURE.md` | `tests/test_world.py::test_grid_shape_and_values`, `tests/test_agent.py::test_pheromone_trail_deposit_and_merge` |
| Operators rationale | Pheromone score, BFS reroute, evaporation, pings, energy-aware rule, PSO velocity update (`swarmrescue/optimizer.py::_Swarm.move`) | `tests/test_agent.py::test_pheromone_rule_picks_unvisited`, `tests/test_evaporation.py::test_evaporation_is_geometric_decay`, `tests/test_pings.py::test_ping_takes_priority_over_pheromone` |
| CI technique rationale | README "Why these CI techniques?" table | `tests/test_optimizer.py::test_never_worse_than_baseline` |
| What-changed notes | README "Attempt Notes", `CHANGELOG.md` | `tests/test_traceability.py::test_changelog_covers_every_attempt` |

## 5. Project brief (algorithm and structure)

| Brief line | Implementation (module :: function) | Verified by (test) |
|---|---|---|
| Grid map: 0 = free, 1 = debris/wall | `swarmrescue/world.py::generate_disaster_zone`, `swarmrescue/world.py::CellType` | `tests/test_world.py::test_cell_types` |
| Each robot decides alone from local information | `swarmrescue/agent.py::Robot.choose_move`, `swarmrescue/agent.py::MoveContext` | `tests/test_agent.py::test_pheromone_rule_picks_unvisited` |
| Pheromone rule: `W_p·visits + W_s·crowding + R·noise` | `swarmrescue/agent.py::Robot._pheromone_rule`, `swarmrescue/agent.py::crowding_many` | `tests/test_agent.py::test_crowding_formula`, `tests/test_agent.py::test_spread_term_avoids_teammates`, `tests/test_efficiency.py::test_vectorised_crowding_matches_scalar_version` |
| BFS fallback to the nearest unvisited known-free cell | `swarmrescue/agent.py::Robot._reroute_to_frontier`, `swarmrescue/agent.py::bfs_path` | `tests/test_agent.py::test_bfs_reroute_when_neighbours_searched`, `tests/test_agent.py::test_bfs_returns_true_shortest_path` |
| Deadlock breaker after 3 ticks | `swarmrescue/agent.py::Robot._break_deadlock` | `tests/test_agent.py::test_deadlock_breaker_after_three_blocked_ticks` |
| Sensing within `SENSE_RANGE` | `swarmrescue/agent.py::Robot.sense_debris` | `tests/test_agent.py::test_sense_updates_window_only` |
| Map sharing within `COMM_RANGE` (`np.maximum`) | `swarmrescue/coordination.py::RadioLink.share_pheromone_trails`, `swarmrescue/agent.py::Robot.absorb` | `tests/test_agent.py::test_radio_link_merges_with_maximum`, `tests/test_efficiency.py::test_radio_merge_uses_absorb` |
| Battery: each move costs 1, and a robot stops at 0 | `swarmrescue/agent.py::Robot.commit`, `swarmrescue/agent.py::Robot.active` | `tests/test_simulation.py::test_battery_limits_energy` |
| Round 2: debris, robot 0 fails, radio drops, reachable area recomputed | `swarmrescue/simulation.py::MissionControl.trigger_aftershock`, `swarmrescue/world.py::drop_aftershock_debris` | `tests/test_simulation.py::test_round2_shift_effects`, `tests/test_world.py::test_aftershock_debris_placed_correctly` |
| Fitness formula | `swarmrescue/simulation.py::compute_fitness` | `tests/test_simulation.py::test_fitness_formula_exact` |
| Maximum decision latency per tick | `swarmrescue/telemetry.py::MissionTelemetry.record_tick` | `tests/test_simulation.py::test_latency_per_tick_under_50ms` |
| PSO: 8 particles, 12 iterations, w=0.6, c1=c2=1.5, seeds 1–3 | `swarmrescue/optimizer.py::run_pso` | `tests/test_optimizer.py::test_best_fitness_never_decreases`, `tests/test_optimizer.py::test_params_within_bounds_and_reproducible` |
| Frozen config dataclass; all randomness seeded | `swarmrescue/config.py::SwarmConfig`, `swarmrescue/simulation.py::MissionControl.__init__` (independent SeedSequence streams) | `tests/test_config.py::test_config_is_frozen`, `tests/test_simulation.py::test_deterministic_with_same_seed` |
| Rule-based Mission Advisor (no API keys) | `swarmrescue/advisor.py::advise`, `swarmrescue/advisor.py::MissionAdvisor` | `tests/test_advisor.py::test_successful_mission`, `tests/test_advisor.py::test_missed_survivors_increase_spread` |
| CLI `--round`, `--tune`, `--seeds` | `main.py::build_parser`, `main.py::main` | `tests/test_cli.py::test_cli_round2_runs`, `tests/test_cli.py::test_cli_bounds_every_input` |
| Streamlit Mission Control | `app.py::main`, `swarmrescue/render.py::render_frame` | `tests/test_app.py::test_dashboard_round2_and_tuning`, `tests/test_app.py::test_dashboard_config_round_trip_with_tuned_weights` |
