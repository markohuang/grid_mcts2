# Why lc0 Scales: Training Logistics, Hyperparameters, and Engineering

An analysis of the specific design decisions that enable Leela Chess Zero to achieve superhuman chess play, and what they reveal about scaling AlphaZero-style systems.

---

## 1. The Core Loop and Its Scale

lc0 follows the AlphaZero loop: **self-play → collect data → train network → repeat**. What makes it work is the sheer volume at every stage.

### Data Generation Scale

| Metric | lc0 (community training) | Typical small-scale project |
|--------|--------------------------|----------------------------|
| Simulations per move | ~800 | 50 |
| Positions per game | ~60 (avg chess game) | ~12 (short episodes) |
| Games per training window | Tens of thousands (continuous) | 20 per epoch |
| Total training positions | **Billions** over project lifetime | ~12,000 total |
| Replay buffer | Millions of positions (external) | 1,000 transitions |

The ratio matters: **~800 simulations across ~1858 legal moves** means roughly 43% of moves get at least one visit. At 50 simulations across ~109 actions, only ~46% get visited — numerically similar, but the tree depth is shallower and the value estimates noisier because fewer backup paths exist.

The real gap is total training data. lc0 trains on **billions** of positions. A small project generating 20 games × 12 steps × 50 epochs = 12,000 transitions is operating with **5–6 orders of magnitude less data**. Neural networks need volume to generalize.

---

## 2. Search Hyperparameters That Enable Quality Data

The quality of self-play data depends entirely on how well MCTS explores. lc0's defaults are carefully tuned:

### PUCT Formula

```
PUCT(move) = Q(move) + cpuct(N) × P(move) × √N_parent / (1 + N_move)

cpuct(N) = 1.745 + 3.894 × log((N + 38739) / 38739)
```

Source: `src/search/classic/params.cc:543-548`, `src/search/classic/search.cc:446-452`

The **logarithmically growing cpuct** is critical. Early in search (few visits), cpuct ≈ 1.745 — the search trusts the network's policy priors heavily. As visits accumulate, cpuct grows, forcing exploration of moves the network undervalues. This adaptive balance means:
- With 100 visits: cpuct ≈ 1.75 (trust the network)
- With 10,000 visits: cpuct ≈ 3.9 (explore more broadly)
- With 100,000 visits: cpuct ≈ 6.0 (heavily explore)

This is why more simulations don't just reduce noise — they qualitatively change what the search finds.

### First Play Urgency (FPU)

```
FPU = -Q(parent) - 0.330 × √(visited_policy_mass)
```

Source: `src/search/classic/search.cc:438-444`

Unvisited moves are assigned a value *slightly worse* than the parent's current estimate, scaled by how much policy mass has already been explored. This prevents both:
- Wasting visits on bad moves (FPU is pessimistic)
- Never trying new moves (FPU gets less pessimistic as good moves are exhausted)

At root, FPU = 1.0 (maximally optimistic) — ensuring breadth at the decision point.

### Dirichlet Noise

```
P_noisy(move) = 0.75 × P_network(move) + 0.25 × Dirichlet(0.3)
```

Source: `src/search/classic/search.cc:201-218`

Applied **only at root, only during training**. The alpha=0.3 with chess's ~30 legal moves produces moderately spiky noise — occasionally a random move gets significant prior mass, forcing the tree to explore it. This is how the system discovers moves the network has not yet learned to value.

---

## 3. Engineering That Enables Scale

Raw hyperparameters are necessary but not sufficient. lc0's engineering allows it to convert hardware into useful simulations efficiently.

### Batched Neural Network Inference

Source: `src/search/classic/search.cc:1285-1434`

MCTS needs one NN evaluation per leaf expansion. Evaluating leaves one-at-a-time wastes GPU throughput. lc0 solves this with `GatherMinibatch()`:

1. Multiple search threads traverse the tree simultaneously
2. Each thread finds a leaf node and queues it for evaluation
3. Once the batch reaches `target_minibatch_size` (typically 256–1024 on GPU), fire a single batched NN call
4. All waiting threads receive their results and backpropagate

This converts sequential per-leaf NN calls into **one batched GPU call per ~256 leaves**, achieving near-peak GPU utilization. Without this, the GPU would sit idle 99% of the time.

### Virtual Loss Enables Parallelism

Source: `src/search/classic/node.h:310-313`

When a thread selects a path through the tree, it increments `n_in_flight` on every node it touches. This acts as a **temporary penalty** — other threads see those nodes as slightly more visited (less attractive under PUCT) and explore elsewhere. When the NN result returns, `n_in_flight` is decremented and the real value is installed.

This is what allows dozens of threads to search simultaneously without a global lock. The tree is never "locked" — threads just avoid each other naturally through virtual loss pressure.

### Tree Reuse Between Moves

Source: `src/search/classic/node.cc:413-449`

After playing a move, lc0 keeps the subtree rooted at the played move and discards all siblings. Empirically, **~52% of the tree survives** on average (tracked in `src/search/classic/stoppers/smooth.cc:49-55`). This means each new search starts with thousands of pre-computed visits rather than an empty tree — effectively getting free simulations.

### Position Cache

Source: `src/neural/memcache.h:34-45`

A hash table caches NN evaluations by position. If the same position is reached via different move orders (transpositions), the cached result is returned instantly. In the DAG variant (`src/search/dag_classic/node.h:500-511`), nodes can have multiple parents, sharing entire subtrees across transpositions.

### Smart Pruning

Source: `src/search/classic/stoppers/stoppers.cc:186-250`

When the best move is so far ahead that no remaining simulations could change the decision, search stops early. This reallocates compute from "already decided" positions to harder ones — effectively increasing simulations-per-move where it matters.

### Backend Diversity

lc0 supports 10+ compute backends: CUDA, cuDNN, OpenCL, Metal, DirectX, SYCL, BLAS (CPU), oneDNN, XLA, TensorFlow. This is how a distributed community project scales — anyone with any GPU can contribute self-play games.

---

## 4. Network Architecture Choices

### WDL Over Scalar Value

Source: `src/neural/network.h:85-114`

The value head outputs **three values (Win, Draw, Loss)** rather than a single Q ∈ [-1, 1]. This matters because:

- A position with 50% win / 50% loss (volatile, tactical) has Q=0
- A position with 100% draw (stable, quiet) also has Q=0
- WDL distinguishes these — the first has D=0, the second D=1

This gives MCTS better signal for choosing between sharp and safe lines. The draw score parameter lets the search tune aggressiveness:

```
effective_Q = (W - L) + draw_score × D
```

### Policy Compression

Source: `src/search/classic/node.h:85-112`

Policy priors are stored as **16-bit floats** (5-bit exponent + 11-bit significand) per edge. With ~30 edges per node and nodes at 64 bytes, the tree fits in cache. This is pure systems engineering — but it's what allows 100M+ node trees in RAM.

### Architecture Evolution

lc0 has progressed through three architecture generations:
- **Classical**: Plain residual tower (baseline)
- **SE**: Squeeze-Excitation blocks (channel attention — significant strength gain)
- **Attention**: Transformer-based body (current strongest)

Up to 384 filters and 30+ residual blocks. The network is stored externally — the engine loads whatever weights file is provided, so architecture experiments don't require rebuilding the engine.

---

## 5. The Distributed Training Pipeline

### Separation of Concerns

lc0 (this repo) handles:
- MCTS search
- Self-play game generation
- Training data serialization (V6TrainingData, 8356 bytes/position)

A separate project (lczero-training) handles:
- Gradient descent (learning rate, optimizer, schedule)
- Replay buffer management
- Network architecture definition and training

This separation is itself a scaling decision — the self-play engine is optimized purely for throughput (C++, GPU backends, lock-free trees), while the trainer is optimized for flexibility (Python, PyTorch/TF).

### Asynchronous Pipeline

The pipeline is **not** synchronous (play N games → train → repeat). Instead:

```
Self-play workers (hundreds of volunteers)
    ↓ upload games continuously
Central server
    ↓ feeds positions to trainer
Trainer
    ↓ publishes new network periodically
Self-play workers download new network → repeat
```

There is no "wait for training to finish" bottleneck. Self-play runs continuously on the current best network while the trainer improves it in the background. This keeps hardware utilized at all times.

### Training Data Format

Source: `src/trainingdata/trainingdata_v6.h:37-85`

Each position records:
```
planes[104]           — board state (input)
probabilities[1858]   — MCTS visit distribution (policy target)
result_q / result_d   — game outcome (value target)
plies_left            — moves to game end (moves-left target)
root_q / best_q / played_q  — search evaluations (metadata)
policy_kld            — KL divergence: network vs search (quality metric)
```

The **policy target is the visit distribution, not the network output**. This is the bootstrap: the network learns to predict what MCTS would do, and MCTS uses the network to guide search. Each is always slightly better than the other.

---

## 6. Why It Scales — The Compound Effects

No single feature explains lc0's strength. The system scales because **improvements compound across all three components**:

### Better Network → Better Search
A more accurate policy prior means MCTS wastes fewer simulations on bad moves. With 800 simulations and a strong network, nearly all visits go to the top 5–10 moves — each getting 80–160 deep evaluations. With a weak network, visits spread across 30+ moves, each getting ~25 shallow evaluations.

### Better Search → Better Training Data
MCTS with 800 simulations corrects network errors through lookahead. The visit distribution is a strictly better policy than the raw network output. When the network trains on this improved target, it absorbs the search's corrections. **The training signal is always slightly ahead of the network.**

### More Data → Better Generalization
Chess has ~10^44 legal positions. With billions of training positions from diverse self-play games (including Dirichlet-noise-forced lines), the network sees enough variety to generalize rather than memorize. The replay buffer ensures old positions remain in the training mix.

### More Hardware → All of the Above
Adding GPUs increases both self-play throughput (more games) and search quality (more simulations per move if desired). The distributed architecture means scaling is linear — double the volunteers, double the game output.

### The Flywheel

```
More simulations per move
  → higher quality visit distributions
    → better policy targets for training
      → stronger network
        → simulations are more efficient (better priors)
          → effectively more simulations per unit of compute
            → [repeat]
```

This is the fundamental reason AlphaZero-style systems scale: **compute converts to data quality, which converts to network quality, which converts back to more efficient compute usage**. The system improves its own ability to use additional resources.

---

## 7. Concrete Numbers

For reference, the scale at which lc0 operates:

| Aspect | Value |
|--------|-------|
| Action space | 1858 possible moves |
| Simulations per move (training) | ~800 |
| Simulations per move (tournament) | 10,000–100,000+ |
| Batch size for NN inference | 256–1024 per GPU call |
| Search threads | 2–8 (typical single-GPU) |
| Tree reuse | ~52% of nodes survive between moves |
| Node size | 64 bytes (cache-line aligned) |
| Training positions generated | Billions (lifetime) |
| Network size | Up to 384 filters, 30+ blocks |
| Training batch size | 4096 (typical trainer config) |
| GPU backends supported | 10+ |
| Self-play workers | Hundreds (distributed community) |
| Pipeline | Asynchronous (no sync barrier) |
