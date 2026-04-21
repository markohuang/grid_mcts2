# Core MCTS Tracker

Use this file as the live queue for experiment planning and decisions.
Move detailed analysis into the round docs once a study is complete.

## Active Hypotheses

| Study | Hypothesis | Status | Primary metric | Decision threshold | Owner | Notes |
|-------|------------|--------|----------------|--------------------|-------|-------|
| `round08_reward_error_modes` | H1: `plan_cost` generalist failure is primarily reward cancellation | partial evidence | held-out avg cost + reward-path stats | materially lower cancellation signal than `layer_delta` under matched settings | marko | cancellation is real, but fixing it with pure `layer_delta` did not improve generalist seed42 |
| `round08_reward_error_modes` | H2: `layer_delta` improves generalist optimization but may be myopic | unsupported on seed42 | held-out avg cost at ep10/30/60 | beat or match `plan_cost` on 2 seeds without later-layer regression | marko | `layer_delta` preserved signal but stayed shallow and weaker than `plan_cost` |
| `round08_reward_error_modes` | H3: current value-calibration conclusions are too noisy | still open | stable `val_cost_corr` from 100+ no-noise eval games | calibration estimate consistent enough to compare rewards | marko | diagnostics added, but calibration evidence still needs larger eval |
| `round09_search_bonus_generalist_5x5` | H1: MCTS-only `plan_cost` bonus restores long-horizon search to `layer_delta` on random 5x5 boards | planned | held-out avg cost + `avg_mcts_depth` + `mcts_boundary_reach_frac` | treatment materially deepens search over plain `layer_delta` | marko | motivated by strong fixed-map result `66559890` |
| `round09_search_bonus_generalist_5x5` | H2: `layer_delta + search_bonus` beats plain `layer_delta` and closes the gap to `plan_cost` | planned | held-out avg cost at ep10/30/60 | treatment beats `layer_delta` and is competitive with `plan_cost` on 2 seeds | marko | tests search-only surrogate as the lowest-risk hybrid |
| `round09_search_bonus_generalist_5x5` | H3: if search bonus only helps specialists, the gain disappears on random boards | planned | held-out avg cost under matched seeds | treatment depth rises but cost stays close to plain `layer_delta` | marko | helps distinguish true structural gain from fixed-map shortcutting |

## Live Runs

| Run ID | Study | Hypothesis | Variant | Status | Expected stop | Notes |
|--------|-------|------------|---------|--------|---------------|-------|
| `e9968d5c` | `round08_reward_error_modes` | H1 | `plan_cost_g` | completed | ep60 | generalist control, seed42 |
| `153cf066` | `round08_reward_error_modes` | H2 | `layer_delta_g` | completed | ep60 | direct reward comparison, seed42 |
| `2bdbe5b8` | `round08_search_bonus_5x5` | H1 | `layer_delta_fixed_control` | completed | ep40 | fixed 5x5 control for search-bonus follow-up |
| `66559890` | `round08_search_bonus_5x5` | H1 | `layer_delta_fixed_search_bonus_w2` | completed | ep40 | fixed 5x5 treatment; large gain over control |
| `e3b850c4` | `round09_search_bonus_generalist_5x5` | H2 | `plan_cost_g_control` | running | ep60 | generalist baseline, seed42 |
| _pending_ | `round09_search_bonus_generalist_5x5` | H1 | `layer_delta_g_control` | queued | ep60 | plain `layer_delta` baseline, seed42 batch |
| _pending_ | `round09_search_bonus_generalist_5x5` | H2 | `layer_delta_g_search_bonus_w2` | queued | ep60 | MCTS-only `plan_cost` bonus, seed42 batch |
| _pending_ | `round08_reward_error_modes` | H1 | `plan_cost_g_s200` | conditional | ep60 | only if B1 is ambiguous |
| _pending_ | `round08_reward_error_modes` | H2 | `layer_delta_g_s200` | conditional | ep60 | only if B1 is ambiguous |

## Decision Ledger

| Date | Study | Hypothesis | Outcome | Decision | Follow-up |
|------|-------|------------|---------|----------|-----------|
| 2026-03-31 | `round08_reward_error_modes` | H1/H2/H3 | planned | run direct generalist reward comparison before changing architecture | add reward-path + stable eval diagnostics |
| 2026-04-02 | `round08_reward_error_modes` | H1/H2 | `plan_cost` beat plain `layer_delta` on the matched seed42 generalist diagnostic, despite much lower `mcts_reward_frac` | do not switch the generalist reward to pure `layer_delta` | test whether `plan_cost` is more useful as search structure than as the learning target |
| 2026-04-02 | `round08_search_bonus_5x5` | H1 | fixed-map search bonus treatment `66559890` strongly beat control `2bdbe5b8` | promote MCTS-only surrogate bonus to the next generalist follow-up | launch round09 three-way comparison on random 5x5 boards |
| 2026-04-02 | `round09_search_bonus_generalist_5x5` | H1/H2/H3 | launched | run the seed42 three-way batch before spending more compute on seed123 | fill in the two queued run IDs as the batch advances |

## Run Launch Checklist

- Hypothesis is written as a falsifiable sentence.
- Control and treatment differ on one main axis.
- Success metric and threshold are written down before launch.
- Run command includes `study`, `hypothesis`, `variant`, and `notes`.
- Planned follow-up decision is written down.
