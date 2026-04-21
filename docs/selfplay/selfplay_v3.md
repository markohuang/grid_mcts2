# Self-play v3: random-map waves

Continuation of `selfplay_waves.md` (v1–v2 on fixed map 2). v3 shifts to random maps every game, so self-play data is drawn from a distribution of problem instances rather than a single point. Fixed map 2 is retained for benchmarking only (separate evaluation script, not training data).

## Motivation (established by v2)

- v2 confirmed that at 10k sims + AlphaDev defaults + FakeNet on map 2: cost median ~23, min 13 (optimum 12), 100% unique trajectories across 20k games. Search is functional at this scale.
- v2 also confirmed that *trajectory* diversity is not a constraint (0% repeats at 20k games), but *state-space coverage* within a fixed start is structurally limited. An actor-critic trained on v2 data would generalize to random-initial-position variants only insofar as the NN is architecturally invariant to initial placement.
- v3 eliminates this confound: each game is an independent problem instance. The NN sees a distribution of starts; generalization is baked in at data-generation time rather than trusted to emerge.

## Assumptions I'm making for v3 (flag any before we implement)

### Hard assumptions — would invalidate the whole plan if wrong

1. **`random_board=True` will produce a different `atom_map` and `tasks` per game.** Via `map_generator.generate_random_map(board_dim, num_qubits, num_layers, gates_per_layer, seed=None)`. Seeds are None → different every call.
2. **Map *structure* (board size, qubit count, layer count, gates per layer) stays fixed.** Only the specific qubit positions and layer gate assignments vary per game. This keeps action space dimensionality constant, which the NN requires.
3. **Per-game maps are approximately as hard as map 2** — because `map_generator` is structure-preserving. But we have no direct validation of this — each random map has an unknown optimum, so "is game X cost 30 good or bad?" becomes unanswerable without a per-map oracle. We won't have per-wave absolute cost targets anymore.
4. **Cost distribution at the wave level is still meaningful** — even though individual game costs aren't comparable to a known optimum, the wave-level distribution across maps reflects "how well does search do on a random instance of this problem class".

### Known gaps — status

5. **`selfplay_worker.py` `random_board` support**: ✅ ported 2026-04-21. `_play_single_game` now regenerates a fresh map per game in the worker process when `config.random_board=True`. `_enrich_game_dict` derives per-game `map_id` from `Game.tasks` + `initial_positions`. Smoke-tested with 6 games → 6 distinct `map_id`s, single shared `map_class`. Parquet index has per-game `map_id`.
6. **`save_game_batch` map-id placeholder**: ✅ done. Added `map_id_override` optional arg. When `random_board=True`, batches land in `games/{map_class}/mixed/{batch}.pt` — keeps per-class directory for filtering without per-game directory explosion.
7. **`save_map_spec` skip on random**: ✅ done. `run_worker` skips per-wave map-spec JSON dump when `random_board=True` (reference map is a template, not an actual map any game uses). Per-game map info lives in the .pt game dicts and the parquet index.
8. **`check_wave_health.py` multi-map support**: ⚠️ NOT yet ported. Current script reports cost stats over the full wave, which on random maps mixes different problem instances. **Still works for cross-wave aggregate comparisons** (e.g. "v3a mean cost vs v3b mean cost") because both waves draw from the same structure-class distribution — just not meaningful for per-game interpretation. Refinement deferred until after first v3 results.

### Soft assumptions — worth noting but wouldn't break things

8. **FakeNet remains ≡ random-init NN for data generation** at the wave scale — established by v2a on fixed map 2, assumed to hold on random maps.
9. **Inheriting v2's best config**: 10k sims, `pb_c_base=19652`, `root_dirichlet_alpha=0.1` (new default), `layer_delta`, `temperature_init=1.0` constant.
10. **Same sample size as later v2 waves**: 1000 games total, 5 jobs × 200 games/node. Scales inversely with random-map variance: if the per-map cost distribution is much wider than on map 2, we may need more games to stabilize the wave-level distribution. Will revisit after v3a.
11. **Random map seeds are truly random** per game (`random_board_seed=-1` → None → system entropy). We don't control for "different jobs happen to regenerate the same map" — at 1000 games across 5 jobs at structure `5x5 / 12qb / 3 layers / 4 gpl`, the number of distinct structure-compatible maps is large enough (~12! for qubit placements × combinatorial tasks) that duplicates are vanishingly unlikely, but we don't actively dedupe.

### Things I'm not testing in v3a

12. **Random map structure** — v3a keeps the same board size / qubit count / layer structure as map 2. Varying structure (e.g. 4x4 with 8qb mixed with 5x5 with 12qb) would require architecture changes (different action space sizes). Deferred.
13. **Curriculum over map difficulty** — config has `curriculum_maps` support but v3a doesn't use it. Deferred.
14. **Priority sampling** — `priority_exponent=0.0` (uniform sampling). Deferred to train_offline.py phase.

## v3a and v3b — first random-map waves (A/B on reward_mode)

After v2j showed plan_cost > layer_delta on fixed map 2, reward_mode became an open question for random maps. v3a and v3b are a clean A/B — identical otherwise.

### Shared config

```
--config.use_fake=True
--config.map_num=2                      # structure template only (5x5 12qb 4gpl 3lyrs)
--config.random_board=True              # freshly-generated map per game
--config.mcts.num_simulations=10000
# pb_c_base=19652, pb_c_init=1.25, dirichlet=0.1 inherited from config.py defaults
# temperature_init=1.0 constant (decay_moves=0) inherited
```

### Shape (each wave)

- 20000 games total
- 20 jobs × 1000 games/node
- `--time=01:30:00`
- Launch: `sbatch slurm/selfplay_v3a.sh` and `sbatch slurm/selfplay_v3b.sh`

### The A/B

| wave | reward_mode | dataset |
|---|---|---|
| v3a | **plan_cost** (v2j winner on map 2) | `wave03a_rnd_plancost` |
| v3b | **layer_delta** (v2f default on map 2) | `wave03b_rnd_layerdelta` |

### What we'll learn

1. **Is cost distribution wider than v2f's map-2-only distribution?** If yes, random maps cover more of the problem-instance distribution, which is what we want for training.
2. **Is `reached_terminal_frac` similar, higher, or lower than v2f's (which we'll measure at 10k sims in v2i as part of the v2 wrap-up)?** Different per-game maps may have different effective depths; this tells us whether "sim count needed for full-depth" generalizes across instances or needs per-map tuning.
3. **Does `corr(root_v, cost)` stay near zero or pick up a sign?** On map 2 the correlation was near-zero due to shallow-depth value-head bias; random-map waves are a cleaner test of this because per-game value targets aren't confounded by a single oracle best.
4. **Per-map cost variance vs within-map cost variance.** This is the critical training-data-quality question: are 1000 games on 1000 different maps more diverse than 1000 games on one map? We hypothesize yes; v3a measures it.

## Prerequisites — status (updated 2026-04-21)

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Port `random_board` in `selfplay_worker.py` | ✅ | `_play_single_game` regenerates per-game; smoke-tested |
| 2 | `save_game_batch` per-game map_id handling | ✅ | `map_id_override='mixed'` path component |
| 3 | `_enrich_game_dict` per-game stamping | ✅ | derived from `Game.tasks` + `initial_positions` |
| 4 | `check_wave_health.py` multi-map support | ⚠️ deferred | aggregate numbers still meaningful for A/B; per-map breakdown after first v3 results |
| 5 | End-to-end smoke test | ✅ | 6-game run produced 6 distinct `map_id`s, 1 `map_class` |

**Ready to launch v3a and v3b.**
