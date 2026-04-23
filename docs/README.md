# Documentation Index

## Start here

- [problem_definition.md](problem_definition.md) — What we're solving: neutral atom reconfiguration to minimize parallel execution cost
- [pipeline.md](pipeline.md) — Training pipeline: features → network → MCTS → training targets
- [architecture.md](architecture.md) — File structure, interfaces, hyperparameters, device lifecycle
- [mcts_selfplay_diagnostics_v2.md](mcts_selfplay_diagnostics_v2.md) — **Current.** MCTS/self-play critical read (HEAD `75a8498`), hyperparameter health checklist, scaling blockers, and logging plan for AC dataset
  - [mcts_selfplay_diagnostics.md](mcts_selfplay_diagnostics.md) — v1 (kept as changelog; v1 flagged issues that were already fixed on HEAD — see v2 §0 for the diff)

## MCTS algorithm proposals

- [gumbel_pczero_plan.md](gumbel_pczero_plan.md) — Gumbel AlphaZero + PCZero literature summary and 2×2 ablation plan (control / A=Gumbel / B=PCZero / AB) against `selfplay_v3.md` baseline

## Self-play waves

- [selfplay/selfplay_v3.md](selfplay/selfplay_v3.md) — v3a/v3b random-map waves; plan_cost vs layer_delta A/B; v3a is the reference baseline
- [selfplay/selfplay_v4.md](selfplay/selfplay_v4.md) — v4 8×8 scaling study
- [selfplay/selfplay_gumbel.md](selfplay/selfplay_gumbel.md) — **Current.** Gumbel AlphaZero ablation waves; Phase 1 smoke + Phase 2 A/B results; open issues (σ formula, policy entropy)

## Reward modes

- [reward_modes.md](reward_modes.md) — **All 4 active reward modes**: plan_cost, layer_delta, layer_delta+search_bonus, layer_completion. Formulas, code snippets, properties, and results
- [reward_analysis.md](reward_analysis.md) — Detailed derivations: why Option A telescopes, Q-collapse proof, correction term math
- [reward_hybrid_designs.md](reward_hybrid_designs.md) — Hybrid reward proposals (layer_delta + plan_cost shaping as potential function)

## Baselines

- [baselines/simulated_annealing.md](baselines/simulated_annealing.md) — SA baseline (Kirkpatrick '83, Fast-SA, Enola) adapted to per-layer atom placement with exact env cost

## Reference

- [alphadev_comparison.md](alphadev_comparison.md) — AlphaDev architecture comparison
- [iterative_refinement_reference.md](iterative_refinement_reference.md) — Iterative refinement approaches
- [map_pool.json](map_pool.json) — 5×5 map pool with best known costs
- [narval_hpc_migration.md](narval_hpc_migration.md) — HPC deployment (Narval cluster)

## Experiments

See [docs/experiments/README.md](./experiments/README.md) for the full round-by-round index.

Key rounds:
- [R07](./experiments/07_generalist_and_reward_modes.md) — Reward mode shootout (plan_cost vs layer_delta vs layer_completion)
- [R08](./experiments/08_reward_signal_error_modes.md) — layer_delta + search bonus discovery
- [R10](./experiments/10_convergence_and_scaling.md) — Convergence studies, sequential specialist, architecture analysis
- [R11](./experiments/11_architecture_validation.md) — **Current**: Transformer + cross-positional features validation
