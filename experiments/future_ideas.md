# Future Experiment Ideas

Unprioritized ideas for improving the neutral atom MCTS system. See `README.md` for current experiment status.

## Reward & Credit Assignment

- **Curriculum learning**: Start with 1 task layer, add layers as the agent improves. Reduces credit assignment horizon — reconfig payoff is immediate with 1 layer.
- **Hindsight experience replay (HER)**: Relabel failed episodes with the goal the agent actually achieved. May help when reconfig moves lead to suboptimal but non-trivial solutions.
- **Potential-based reward shaping**: Use gate-only cost as a potential function φ(s). Reward = γφ(s') - φ(s) + environment reward. Provably preserves optimal policy while providing denser signal.

## Search Efficiency

- **Batched MCTS inference**: Collect leaf nodes across multiple simulations, batch network forward passes. Current bottleneck is sequential network calls per simulation.
- **Progressive widening**: For large action spaces (109 actions on map 0), limit children at each node to C * N^α where N = visit count. Focuses search on promising actions.
- **Action space pruning**: Restrict moves to neighboring cells or cells adjacent to gate partners. Reduces branching factor from O(Q*B) to O(Q*4) or O(Q*P).

## Network & Representation

- **MuZero-style learned dynamics**: Replace env cloning in MCTS with a learned transition model. Enables deeper search without expensive env simulation.
- **Symmetry-aware data augmentation**: Board has spatial symmetries (rotation, reflection). Augment training data by applying symmetry transforms to (obs, policy) pairs.
- **GNN for atom interactions**: Replace MLP with graph neural network where atoms are nodes and gate pairs are edges. May generalize better to different maps.

## Plan Modification / Local Search

Unlike chess or assembly (AlphaDev) where a mid-plan change invalidates everything after it, atom reconfig plans have a useful property: **inserting or removing a move mid-plan doesn't invalidate the suffix** — atoms are still on the board, the remaining moves/executes are still valid (just potentially suboptimal). This opens up approaches that don't exist in standard AlphaZero domains:

- **Post-MCTS local search**: After MCTS finds a complete solution, hill-climb by trying single-move insertions/deletions/swaps at each step. Pure env evaluation, no network needed. Cheap refinement layer.
- **Plan surgery actions**: Extend the action space to include "insert move before step K" — converts sequential planning into local search. Challenge: action space definition and compatibility with MCTS tree structure.
- **Bidirectional planning**: For each gate layer, compute ideal qubit positions (all gate pairs adjacent), then plan reconfig moves to reach that configuration. Decomposes into independent per-layer subproblems with shorter horizons.
- **Experience replay with plan mutations**: Take best games, randomly insert/remove reconfig moves, replay through env for new rewards, add as augmented training data.

## Training

- **Population-based training (PBT)**: Run multiple agents with different hyperparameters, periodically copy weights from best performers. Auto-tunes entropy weight, lr, etc.
- **Self-play curriculum**: After solving easy maps, use solutions as demonstrations for harder maps. Transfer learning across map sizes.
- **Off-policy correction**: Current replay buffer mixes data from different policy versions. Importance sampling or V-trace could improve sample efficiency.
