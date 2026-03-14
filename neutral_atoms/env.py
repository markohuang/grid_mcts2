import torch
from .types import StepResult, EMPTY_CELL, to_json
from .board import create_board, move_qubit_to_cell
from .tasks import is_episode_done, get_relevant_atoms
from .rewards import compute_total_cost, compute_current_layer_cost, compute_reward, compute_cost_bounds


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
        self.reset()

    def _precompute_relevant_atoms(self):
        self.layer_relevant_atoms = [get_relevant_atoms(self.tasks, i) for i in range(self.num_tasks)]
        self.episode_length = sum(len(atoms) for atoms in self.layer_relevant_atoms)

    def reset(self) -> dict:
        self.board, self.atom_positions = create_board(
            self.config.board_height, self.config.board_width, self.initial_positions
        )
        self.actions = []
        self.tasks_done = 0
        self.current_atom_idx = 0
        self.current_phase_moves = []
        self.total_move_distance = 0
        self._cached_cost = compute_total_cost(
            self.board, self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        self._current_layer_cost = compute_current_layer_cost(
            self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        return self._get_observation()

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

    def step(self, action: int) -> StepResult:
        prev_layer_cost = self._current_layer_cost
        qubit_idx = self.current_qubit
        current_flat = (self.atom_positions[qubit_idx][0] * self.board_width +
                        self.atom_positions[qubit_idx][1]).item()

        if action != current_flat:
            dst_row, dst_col = action // self.board_width, action % self.board_width
            src_row, src_col = self.atom_positions[qubit_idx].tolist()
            self.total_move_distance += abs(dst_row - src_row) + abs(dst_col - src_col)
            self.board, self.atom_positions, move = move_qubit_to_cell(
                self.board, self.atom_positions, qubit_idx, action, self.board_width
            )
            self.current_phase_moves.append(move)

        self.actions.append(action)
        self.current_atom_idx += 1

        # Compute reward from current layer cost BEFORE auto-execute
        curr_layer_cost = compute_current_layer_cost(
            self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        reward = compute_reward(prev_layer_cost, curr_layer_cost, self.reward_scale)

        # Auto-execute layer when all relevant atoms have been placed
        if self.current_atom_idx >= len(self.relevant_atoms):
            self.tasks_done += 1
            self.current_atom_idx = 0
            self.current_phase_moves = []

        self._cached_cost = compute_total_cost(
            self.board, self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        self._current_layer_cost = compute_current_layer_cost(
            self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        done = is_episode_done(self.tasks_done, self.num_tasks)

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={'cost': self._cached_cost, 'tasks_done': self.tasks_done,
                  'cost_lb': self.cost_lb, 'cost_ub': self.cost_ub,
                  'current_qubit': self.current_qubit,
                  'total_move_distance': self.total_move_distance}
        )

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

    def clone(self) -> 'NeutralAtomsEnv':
        new_env = NeutralAtomsEnv(self.tasks, self.initial_positions, self.config)
        new_env.board = self.board.clone()
        new_env.atom_positions = self.atom_positions.clone()
        new_env.actions = self.actions.copy()
        new_env.tasks_done = self.tasks_done
        new_env.current_atom_idx = self.current_atom_idx
        new_env.current_phase_moves = [m.clone() for m in self.current_phase_moves]
        new_env.total_move_distance = self.total_move_distance
        new_env._cached_cost = self._cached_cost
        new_env._current_layer_cost = self._current_layer_cost
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
            'cost': self._cached_cost,
            'legal_actions': self.legal_actions(),
            'done': is_episode_done(self.tasks_done, self.num_tasks),
        }
