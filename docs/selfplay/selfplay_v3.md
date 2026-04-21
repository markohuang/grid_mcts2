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

### Known gaps — need fixing before v3a runs

5. **`selfplay_worker.py` does NOT currently implement `random_board`.** Verified at `selfplay_worker.py:run_worker` — it calls `get_map_data(config)` once and reuses the same `tasks` / `initial_positions` for every game in the wave. The `random_board` switch is only honored in the in-process `trainer._get_game_instance` path (`trainer.py:139-166`). **This must be ported** before v3a can launch. The port is small (~20 lines), but it changes the per-worker game-loop structure.
6. **`map_id` / `map_class` stamping on per-game dicts works with varying maps.** Currently `save_game_batch` stamps a single `map_id` per batch directory based on one map. This likely needs changing so each game has its own `map_id` in the parquet index and the batch storage is map-agnostic (one flat directory, not `games/{map_class}/{map_id}/...`). **Also needs porting**; size ~30 lines.
7. **`check_wave_health.py` assumes single-map data** for cost comparisons. On a multi-map wave, absolute cost across all games mixes apples with oranges. Need to report either per-map stats or cost-relative-to-per-map-baseline. Roughly 1 hour of work.

### Soft assumptions — worth noting but wouldn't break things

8. **FakeNet remains ≡ random-init NN for data generation** at the wave scale — established by v2a on fixed map 2, assumed to hold on random maps.
9. **Inheriting v2's best config**: 10k sims, `pb_c_base=19652`, `root_dirichlet_alpha=0.1` (new default), `layer_delta`, `temperature_init=1.0` constant.
10. **Same sample size as later v2 waves**: 1000 games total, 5 jobs × 200 games/node. Scales inversely with random-map variance: if the per-map cost distribution is much wider than on map 2, we may need more games to stabilize the wave-level distribution. Will revisit after v3a.
11. **Random map seeds are truly random** per game (`random_board_seed=-1` → None → system entropy). We don't control for "different jobs happen to regenerate the same map" — at 1000 games across 5 jobs at structure `5x5 / 12qb / 3 layers / 4 gpl`, the number of distinct structure-compatible maps is large enough (~12! for qubit placements × combinatorial tasks) that duplicates are vanishingly unlikely, but we don't actively dedupe.

### Things I'm not testing in v3a

12. **Random map structure** — v3a keeps the same board size / qubit count / layer structure as map 2. Varying structure (e.g. 4x4 with 8qb mixed with 5x5 with 12qb) would require architecture changes (different action space sizes). Deferred.
13. **Curriculum over map difficulty** — config has `curriculum_maps` support but v3a doesn't use it. Deferred.
14. **Priority sampling** — `priority_exponent=0.0` (uniform sampling). Deferred to train_offline.py phase.

## v3a — first random-map wave

### Config (to be exact)

```
--config.use_fake=True
--config.map_num=2                    # determines structure template (5x5 12qb 4gpl 3lyrs)
--config.random_board=True            # NEW — wave per-game randomizer
--config.mcts.num_simulations=10000
--config.mcts.root_dirichlet_alpha=0.1  # new default (adopted from v2h)
--config.env.reward_mode=layer_delta
# pb_c_base=19652, pb_c_init=1.25, tau=1 constant inherited from config.py defaults
```

### Shape

- 1000 games total
- 5 jobs × 200 games/node
- `--time=00:30:00` (same as v2f/v2j at 10k sims)
- Dataset: `/project/rrg-aspuru/huang651/grid_mcts2/datasets/wave03a_rnd`

### What we'll learn

1. **Is cost distribution wider than v2f's map-2-only distribution?** If yes, random maps cover more of the problem-instance distribution, which is what we want for training.
2. **Is `reached_terminal_frac` similar, higher, or lower than v2f's (which we'll measure at 10k sims in v2i as part of the v2 wrap-up)?** Different per-game maps may have different effective depths; this tells us whether "sim count needed for full-depth" generalizes across instances or needs per-map tuning.
3. **Does `corr(root_v, cost)` stay near zero or pick up a sign?** On map 2 the correlation was near-zero due to shallow-depth value-head bias; random-map waves are a cleaner test of this because per-game value targets aren't confounded by a single oracle best.
4. **Per-map cost variance vs within-map cost variance.** This is the critical training-data-quality question: are 1000 games on 1000 different maps more diverse than 1000 games on one map? We hypothesize yes; v3a measures it.

## Prerequisites before launching v3a

In order:

1. **Port `random_board` support into `selfplay_worker.py`.** Roughly: wrap `_play_single_game` call-site so that per-game, if `config.random_board`, regenerate a fresh map via `generate_random_map` and use those `tasks` + `initial_positions` instead of the shared ones.
2. **Make `data.save_game_batch` handle per-game `map_id`.** Either flatten the directory layout (all batches in one dir, `map_id` tracked in the game dict/index) or include multiple `map_id` subdirectories per batch. The flat layout is simpler.
3. **Stamp `map_id` / `map_class` per-game in `_enrich_game_dict`.** Already happens (lines 100-101 in selfplay_worker) — just need to make sure each game's map is correctly threaded.
4. **Update `check_wave_health.py` for multi-map aggregation.** At minimum: group by `map_id`, report within-map cost distributions + cross-map aggregate. Add a column `num_distinct_maps` to the summary.
5. **Smoke test on a 10-game run.** Verify random maps generate, different `map_id`s land in the index, cost distributions look plausible.

Estimated total implementation time: 3–4 hours, mostly in (1) and (4).

Will not write the v3a slurm script until all five prerequisites land. Will not implement any prerequisite until we've aligned on whether v3 is the right direction (vs e.g. warming up a value head first, or porting more trainer features).
