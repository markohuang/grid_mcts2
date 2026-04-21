# lc0 Temperature Design — Reference Notes

How lc0 actually handles self-play exploration/exploitation, extracted from source. Intended as design reference for our synchronous training loop, where AlphaDev's `training_steps`-gated schedule doesn't fit (workers load a checkpoint once, `training_steps` doesn't drift across a selfplay wave).

Companion to `docs/lc0_findings.md` (which covers lc0's broader scaling story). This doc zooms in on temperature and the mechanisms that give lc0 self-play data its E/E diversity without a central schedule.

---

## 1. Core mechanism — per-move linear decay within each episode

`src/search/classic/search.cc:691–711` inside `EnsureBestMoveKnown`, called **once per move** (not per simulation):

```cpp
float temperature = params_.GetTemperature();
const int cutoff_move = params_.GetTemperatureCutoffMove();
const int decay_delay_moves = params_.GetTempDecayDelayMoves();
const int decay_moves = params_.GetTempDecayMoves();
const int moves = played_history_.Last().GetGamePly() / 2;

if (cutoff_move && (moves + 1) >= cutoff_move) {
  temperature = params_.GetTemperatureEndgame();
} else if (temperature && decay_moves) {
  if (moves >= decay_delay_moves + decay_moves) {
    temperature = 0.0;
  } else if (moves >= decay_delay_moves) {
    temperature *= float(decay_delay_moves + decay_moves - moves) / decay_moves;
  }
  if (temperature < params_.GetTemperatureEndgame()) {
    temperature = params_.GetTemperatureEndgame();
  }
}
```

Three-phase envelope inside every single episode:
1. Moves `[0, decay_delay_moves)`: T = `temperature_init` (constant exploration plateau).
2. Moves `[decay_delay_moves, decay_delay_moves + decay_moves)`: T decays linearly to 0.
3. Moves `≥ decay_delay_moves + decay_moves`: T = 0 (pure argmax), or `temperature_endgame` if set.
4. Orthogonal hard cutoff: `moves + 1 ≥ temp_cutoff_move` forces `temperature_endgame` regardless.

**The key design choice is `moves`, not `training_steps`.** Every game re-runs this envelope from move 0. Network weights change between games but the search schedule is purely intra-episode.

## 2. Sampling formula — normalized visits with cutoffs

`GetBestRootChildWithTemperature` (`src/search/classic/search.cc:832–875`):

```cpp
sum += std::pow(
    std::max(0.0f,
             (max_n <= 0.0f ? edge.GetP()
                            : (edge.GetN() + offset) / max_n)),
    1 / temperature);
```

Details:

- **Sampling weight ∝ (N / N_max)^(1/T)**, not the `softmax(log N / T)` form AlphaDev uses. At T=1 both give the same visit-proportional distribution; at T<1 lc0's power formulation is slightly sharper because it's multiplicative instead of additive in log-space.
- **`temp-value-cutoff`** (`kTemperatureWinpctCutoffId`, default 100): any child whose Q is worse than `max_eval − cutoff/50` is zero-weighted regardless of T. Protects against temperature-sampled catastrophic blunders. This is critical: it decouples "sample broadly among good moves" from "risk playing a known-bad move."
- **`temp-visit-offset`** (default 0): additive offset on visit counts before the power. Positive pushes toward uniform; negative prunes low-visit moves; negative-enough can eliminate moves entirely.
- **Fallback at zero search**: `max_n <= 0` uses raw policy prior `P(a)` instead — temperature acts on the network output directly when no search has happened.

## 3. Default values in the lc0 codebase

All code defaults assume **match-play posture** (tournament / evaluation), not training:

| Param | Code default | Meaning |
|---|---|---|
| `temperature` | 0.0 | argmax, no sampling |
| `tempdecay-moves` | 0 | decay disabled |
| `tempdecay-delay-moves` | 0 | no delay |
| `temp-cutoff-move` | 0 | cutoff disabled |
| `temp-endgame` | 0.0 | |
| `temp-value-cutoff` | 100 | no Q-pruning |
| `temp-visit-offset` | 0.0 | |
| `noise-epsilon` | 0.0 | no Dirichlet in match |
| `noise-alpha` | 0.3 | |
| `cpuct` | 1.745 | |
| `cpuct-base` | 38739 | |
| `cpuct-factor` | 3.894 | |
| `fpu-value` | 0.330 | First Play Urgency reduction |
| `fpu-value-at-root` | 1.0 | maximally optimistic at root |

Source: `src/search/classic/params.cc:543–568`.

**Training runs override all of the above via UCI options on every worker invocation.** There is no training-specific default config committed to the repo — each contributor runs with their own settings, and the aggregate community pool is what the trainer sees.

## 4. Typical community-training settings

Reconstructed from lc0 training tournaments and `FINDINGS.md`-style notes (no canonical config file):

| Param | Typical training | Rationale |
|---|---|---|
| `temperature` | 1.0 | AlphaZero-standard visit-proportional |
| `tempdecay-moves` | 30 | decay over first ~half of an avg chess game |
| `tempdecay-delay-moves` | 0 | decay starts immediately |
| `temp-cutoff-move` | 0 or late (~60) | optional hard switch |
| `temp-endgame` | 0.0 | argmax once decayed |
| `noise-epsilon` | 0.25 | Dirichlet mix at root |
| `noise-alpha` | 0.3 | |
| `cpuct*` family | stock defaults | |

Net effect per ~60-move chess game: moves 0–29 sample with T dropping 1.0 → 0.0, moves 30+ are pure argmax. Every root gets Dirichlet(0.3) at ε=0.25.

## 5. Design decisions that matter for synchronous adaptation

Three load-bearing choices separate lc0 from AlphaDev:

### (A) No `training_steps` coupling anywhere

Every temperature decision is episode-internal. Weights change between games (loaded from file), but the search schedule is purely per-game. This decouples "which weights are live" from "how should we sample" — a cleaner separation than AlphaDev's `visit_softmax_temperature_fn(training_steps)` which bakes training-loop state into the search path.

Implication for us: with workers that load a checkpoint once and run a whole wave, a `training_steps`-gated schedule produces data at a single T. An episode-gated schedule produces data spanning the full E/E envelope regardless of checkpoint lineage.

### (B) Heterogeneous volunteers inject T-diversity structurally

The training network aggregates data from hundreds of volunteers, each picking their own temperature/noise/cpuct settings. No central schedule coordinates this. The aggregate replay buffer naturally spans the E/E spectrum every training step. It's not a designed sweep — it's a property of the distributed topology.

Implication for us: without async volunteers, the equivalent is either (i) per-move decay within each game (covers E/E per episode), or (ii) per-game T-bucketing within a batch (covers E/E per batch). lc0 gets both in practice; we'd explicitly choose one.

### (C) Within any single game, E/E spectrum is already covered

`temperature=1.0` + `tempdecay-moves=30` means every episode exposes exploration (T high, early moves) AND exploitation (T=0, late moves). A single game is self-contained data diversity: the replay buffer doesn't need cross-game accounting to cover both regimes.

Implication: the question "how do we ensure the buffer spans E/E" can be answered at the single-game level. No batch- or epoch-level machinery required.

## 6. Mapping onto our (AlphaDev-derived) codebase

Our `get_temperature(training_steps, config)` in `neutral_atoms/mcts.py:16–23` is AlphaDev-literal: linear interp from `temperature_init` to `temperature_final` over `temperature_decay_steps` gradient updates. For synchronous selfplay where `training_steps` stays fixed across a wave, this produces a constant T for the entire wave.

A lc0-style port would change the argument from `training_steps` to `move_in_episode`:

```python
def get_temperature_episodic(move_in_episode, config):
    T_init = config.temperature_init
    delay  = config.temperature_decay_delay_moves  # NEW knob
    decay  = config.temperature_decay_moves        # NEW knob (renamed from _steps)
    T_end  = config.temperature_final              # reused
    cutoff = config.temperature_cutoff_move        # NEW knob

    if cutoff and move_in_episode + 1 >= cutoff:
        return T_end
    if T_init and decay:
        if move_in_episode >= delay + decay:
            return T_end
        if move_in_episode >= delay:
            T = T_init * (delay + decay - move_in_episode) / decay
            return max(T, T_end)
    return T_init
```

Call site change in `mcts.py:179`: pass `len(game.history)` instead of `training_steps`.

For our 24-step map2 episodes, reasonable starting values would be `T_init=1.0, delay=0, decay=8, T_end=0.0`: moves 0–7 decay 1.0 → 0.125, moves 8–23 are argmax. This exposes every training target to both exploration and exploitation within each game.

## 7. What lc0 does NOT do

For honesty — things lc0 explicitly avoids that you might assume it does:

- **No training_steps-based T schedule.** At all. lc0 has no concept of "training iteration" inside the search engine. All long-horizon scheduling happens upstream in the trainer, which just swaps the weights file.
- **No per-simulation temperature.** Temperature applies only at the action-selection step (end of MCTS), never inside UCB during tree search.
- **No coupling between T and Dirichlet noise schedule.** `noise-epsilon` is a single scalar not modulated by move number or training state. (Exception: some volunteers disable noise past certain move — but via `temp-cutoff-move`-style logic on a fork, not stock.)
- **No adaptive T based on search disagreement** (e.g. KL divergence between policy and visits). Considered and rejected in community discussion as coupling that'd need careful retuning per architecture.

## 8. Takeaways for our next experiments

1. **If the goal is to replicate AlphaDev faithfully**, keep `training_steps`-gated T and fix `temperature_init=2.0 → 1.0`. Accept that one-shot selfplay workers produce single-T data per wave.
2. **If the goal is to get the most training signal per selfplay game in a synchronous loop**, port the lc0 per-move schedule. `get_temperature` takes `move_in_episode` instead of `training_steps`, and every game delivers E/E diversity on its own.
3. **If neither closes the gap to do-nothing baseline at random init**, the bottleneck is the value head rather than sampling. No temperature policy saves a search whose leaf values are all noise — lc0's design assumes the network has non-trivial signal (enforced by the trainer side of the flywheel).

Temperature is the cheap knob. Once it's right, the expensive question (value-head bootstrapping for random-init selfplay) becomes the real frontier.
