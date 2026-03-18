import torch
from .types import StepResult, EMPTY_CELL, to_json
from .board import create_board, move_qubit_to_cell
from .tasks import is_episode_done, get_relevant_atoms
from .rewards import compute_total_cost, compute_current_layer_cost, compute_reward, compute_cost_bounds
from .fast_moves import count_groups_fast


class NeutralAtomsEnv:
    def __init__(self, tasks, initial_positions, config):
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.config = config
        self.board_shape = (config.board_height, config.board_width)
        self.board_width = config.board_width
        self.board_size = config.board_height * config.board_width
        self.num_qubits = config.num_qubits
        self.num_tasks = len(tasks)
        self.cost_lb, self.cost_ub = compute_cost_bounds(tasks)
        self.reward_scale = config.reward_scale
        self._precompute_relevant_atoms()
        self._precompute_gate_indicators()
        self.reset()

    def _precompute_relevant_atoms(self):
        self.layer_relevant_atoms = [get_relevant_atoms(self.tasks, i) for i in range(self.num_tasks)]
        self.episode_length = sum(len(atoms) for atoms in self.layer_relevant_atoms)

    def _precompute_gate_indicators(self):
        """Precompute per-layer gate indicator vectors: (num_tasks, num_qubits).
        gate_indicators[t, q] = 1.0 if qubit q participates in any gate pair in layer t.
        """
        gi = torch.zeros(self.num_tasks, self.num_qubits)
        for t, layer in enumerate(self.tasks):
            for q1, q2 in layer:
                gi[t, q1] = 1.0
                gi[t, q2] = 1.0
        self._gate_indicators = gi
        # Precompute gate pair index tensors for fast gates_to_moves
        self._gate_pair_indices = []
        for layer in self.tasks:
            if layer:
                pairs = torch.tensor(layer, dtype=torch.long)  # (G, 2)
                self._gate_pair_indices.append(pairs)
            else:
                self._gate_pair_indices.append(None)

    def reset(self) -> dict:
        self.board, self.atom_positions = create_board(
            self.config.board_height, self.config.board_width, self.initial_positions
        )
        self.actions = []
        self.tasks_done = 0
        self.current_atom_idx = 0
        self.current_phase_moves = []
        self.total_move_distance = 0
        self.num_moves = 0
        self._cost_dirty = True  # lazy compute _cached_cost only when needed
        self._cached_cost = None
        self._init_board_feat()
        return self._get_observation()

    def _init_board_feat(self):
        """Build board_feat (board_size, num_qubits) from current board state."""
        flat_board = self.board.flatten().long()
        self._board_feat = torch.zeros(self.board_size, self.num_qubits)
        for i in range(self.board_size):
            q = flat_board[i].item()
            if q != EMPTY_CELL:
                self._board_feat[i, q] = 1.0

    @property
    def relevant_atoms(self):
        if self.tasks_done >= self.num_tasks:
            return []
        return self.layer_relevant_atoms[self.tasks_done]

    @property
    def current_qubit(self):
        atoms = self.relevant_atoms
        if self.current_atom_idx < len(atoms):
            return atoms[self.current_atom_idx]
        return -1

    def _get_total_cost(self):
        if self._cost_dirty:
            self._cached_cost = compute_total_cost(
                self.board, self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
            )
            self._cost_dirty = False
        return self._cached_cost

    def step(self, action: int, skip_obs: bool = False) -> StepResult:
        qubit_idx = self.current_qubit
        current_flat = (self.atom_positions[qubit_idx][0] * self.board_width +
                        self.atom_positions[qubit_idx][1]).item()

        if action != current_flat:
            dst_row, dst_col = action // self.board_width, action % self.board_width
            src_row, src_col = self.atom_positions[qubit_idx].tolist()
            self.total_move_distance += abs(dst_row - src_row) + abs(dst_col - src_col)
            self.num_moves += 1
            # In-place board + atom_positions update (no defensive cloning)
            self.board[src_row, src_col] = EMPTY_CELL
            self.board[dst_row, dst_col] = qubit_idx
            self.atom_positions[qubit_idx, 0] = dst_row
            self.atom_positions[qubit_idx, 1] = dst_col
            # Update cached board_feat incrementally
            self._board_feat[current_flat, qubit_idx] = 0.0
            self._board_feat[action, qubit_idx] = 1.0
            move = torch.tensor([src_row, src_col, dst_row, dst_col], dtype=torch.long)
            self.current_phase_moves.append(move)

        self.actions.append(action)
        self.current_atom_idx += 1

        # Option C: reward only at layer completion
        reward = 0.0
        if self.current_atom_idx >= len(self.relevant_atoms):
            # Layer just completed — compute actual cost of this layer
            layer_cost = self._compute_layer_cost_fast()
            reward = -layer_cost * self.reward_scale
            self.tasks_done += 1
            self.current_atom_idx = 0
            self.current_phase_moves = []

        self._cost_dirty = True
        done = is_episode_done(self.tasks_done, self.num_tasks)

        return StepResult(
            observation=None if skip_obs else self._get_observation(),
            reward=reward,
            done=done,
            info={'tasks_done': self.tasks_done,
                  'cost_lb': self.cost_lb, 'cost_ub': self.cost_ub,
                  'current_qubit': self.current_qubit,
                  'total_move_distance': self.total_move_distance,
                  'num_moves': self.num_moves}
        )

    def _compute_layer_cost_fast(self) -> int:
        """Inline layer cost using precomputed gate pair indices."""
        if self.tasks_done >= self.num_tasks:
            return 0
        total = 0
        if len(self.current_phase_moves) > 0:
            reconfig_moves = torch.stack(self.current_phase_moves)
            total += count_groups_fast(reconfig_moves, canonicalize=False)
        pairs = self._gate_pair_indices[self.tasks_done]
        if pairs is not None:
            # Gather: (G, 2, 2) -> (G, 4) gate moves via precomputed indices
            gate_moves = self.atom_positions[pairs].reshape(-1, 4)
            total += 2 * count_groups_fast(gate_moves, canonicalize=True)
        return total

    def _get_observation(self) -> dict:
        flat_board = self.board.clone().flatten().long()
        flat_board[flat_board == EMPTY_CELL] = self.num_qubits
        board_onehot = torch.zeros(len(flat_board), self.num_qubits + 1)
        board_onehot.scatter_(1, flat_board.unsqueeze(1), 1)
        return {
            'board': self.board.clone(),
            'atom_positions': self.atom_positions.clone(),
            'board_onehot': board_onehot,
            'tasks_done': self.tasks_done,
            'current_qubit': self.current_qubit,
            'current_atom_idx': self.current_atom_idx,
        }

    def legal_actions(self) -> list[int]:
        if self.tasks_done >= self.num_tasks:
            return []
        from .board import get_legal_actions_for_qubit
        return get_legal_actions_for_qubit(self.board, self.atom_positions, self.current_qubit)

    def get_features(self) -> torch.Tensor:
        """Build feature tensor from cached board_feat + precomputed gate indicators.
        Returns: (num_tasks+1, board_size, num_qubits) tensor.
        """
        # Single expand+clone instead of num_tasks+1 separate clones
        features = self._board_feat.unsqueeze(0).expand(
            self.num_tasks + 1, -1, -1
        ).clone()
        # Vectorized gate indicator addition for remaining layers
        if self.tasks_done < self.num_tasks:
            # gate_indicators[tasks_done:] has shape (remaining, num_qubits)
            # broadcast across board_size dimension
            features[self.tasks_done + 1:] += self._gate_indicators[self.tasks_done:].unsqueeze(1)
        return features

    def clone(self) -> 'NeutralAtomsEnv':
        """Fast clone: skip __init__/reset, share immutable state, clone mutable."""
        new_env = object.__new__(NeutralAtomsEnv)
        # Shared immutable state (no copy needed)
        new_env.tasks = self.tasks
        new_env.initial_positions = self.initial_positions
        new_env.config = self.config
        new_env.board_shape = self.board_shape
        new_env.board_width = self.board_width
        new_env.board_size = self.board_size
        new_env.num_qubits = self.num_qubits
        new_env.num_tasks = self.num_tasks
        new_env.cost_lb = self.cost_lb
        new_env.cost_ub = self.cost_ub
        new_env.reward_scale = self.reward_scale
        new_env.layer_relevant_atoms = self.layer_relevant_atoms
        new_env.episode_length = self.episode_length
        new_env._gate_indicators = self._gate_indicators  # immutable precomputed
        new_env._gate_pair_indices = self._gate_pair_indices  # immutable precomputed
        # Mutable state (must clone)
        new_env.board = self.board.clone()
        new_env.atom_positions = self.atom_positions.clone()
        new_env._board_feat = self._board_feat.clone()
        new_env.actions = self.actions.copy()
        new_env.tasks_done = self.tasks_done
        new_env.current_atom_idx = self.current_atom_idx
        new_env.current_phase_moves = [m.clone() for m in self.current_phase_moves]
        new_env.total_move_distance = self.total_move_distance
        new_env.num_moves = self.num_moves
        new_env._cost_dirty = True
        new_env._cached_cost = None
        return new_env

    def state_hash(self) -> int:
        return hash((self.atom_positions.tobytes(), self.tasks_done, self.current_atom_idx))

    def to_json(self) -> dict:
        return {
            'board': to_json(self.board),
            'atom_positions': to_json(self.atom_positions),
            'tasks': self.tasks,
            'tasks_done': self.tasks_done,
            'current_qubit': self.current_qubit,
            'current_atom_idx': self.current_atom_idx,
            'actions': self.actions,
            'cost': self._get_total_cost(),
            'legal_actions': self.legal_actions(),
            'done': is_episode_done(self.tasks_done, self.num_tasks),
        }
