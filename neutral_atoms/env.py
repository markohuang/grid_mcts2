import torch
from .types import Tasks, StepResult, GATE_ACTION, EMPTY_CELL, to_json
from .board import create_board, apply_move, action_to_move, get_legal_actions
from .tasks import is_episode_done
from .rewards import compute_total_cost, compute_reward, compute_cost_bounds

class NeutralAtomsEnv:
    def __init__(self, tasks, initial_positions, config):
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.config = config
        self.board_shape = (config.board_height, config.board_width)
        self.num_qubits = config.num_qubits
        self.num_tasks = len(tasks)
        self.cost_lb, self.cost_ub = compute_cost_bounds(tasks)
        self.reset()
    
    def reset(self) -> dict:
        self.board, self.atom_positions = create_board(
            self.config.board_height, self.config.board_width, self.initial_positions
        )
        self.actions = []
        self.tasks_done = 0
        self.current_phase_moves = []
        self._cached_cost, self._cached_entropy = compute_total_cost(
            self.board, self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        return self._get_observation()
    
    def step(self, action: int) -> StepResult:
        prev_cost, prev_entropy = self._cached_cost, self._cached_entropy
        
        if action == GATE_ACTION:
            self.tasks_done += 1
            self.current_phase_moves = []
        else:
            move = action_to_move(action, self.atom_positions, self.board_shape)
            self.board, self.atom_positions = apply_move(self.board, self.atom_positions, move)
            self.current_phase_moves.append(move)
        
        self.actions.append(action)
        self._cached_cost, self._cached_entropy = compute_total_cost(
            self.board, self.atom_positions, self.tasks, self.tasks_done, self.current_phase_moves
        )
        
        reward = compute_reward(
            prev_cost, self._cached_cost, prev_entropy, self._cached_entropy,
            self.config.entropy_weight, self.config.reward_scale
        )
        done = is_episode_done(self.tasks_done, self.num_tasks) or len(self.actions) >= self.config.budget
        
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={'cost': self._cached_cost, 'entropy': self._cached_entropy, 
                  'tasks_done': self.tasks_done, 'cost_lb': self.cost_lb, 'cost_ub': self.cost_ub}
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
        }
    
    def legal_actions(self) -> list[int]:
        if len(self.actions) >= self.config.budget or self.tasks_done >= self.num_tasks:
            return []
        return get_legal_actions(self.board, self.atom_positions, self.num_qubits, include_gate=True)
    
    def clone(self) -> 'NeutralAtomsEnv':
        new_env = NeutralAtomsEnv(self.tasks, self.initial_positions, self.config)
        new_env.board = self.board.clone()
        new_env.atom_positions = self.atom_positions.clone()
        new_env.actions = self.actions.copy()
        new_env.tasks_done = self.tasks_done
        new_env.current_phase_moves = [m.clone() for m in self.current_phase_moves]
        new_env._cached_cost = self._cached_cost
        new_env._cached_entropy = self._cached_entropy
        return new_env
    
    def state_hash(self) -> int:
        return hash((self.atom_positions.tobytes(), self.tasks_done, len(self.current_phase_moves)))
    
    def to_json(self) -> dict:
        return {
            'board': to_json(self.board),
            'atom_positions': to_json(self.atom_positions),
            'tasks': self.tasks,
            'tasks_done': self.tasks_done,
            'actions': self.actions,
            'cost': self._cached_cost,
            'entropy': self._cached_entropy,
            'legal_actions': self.legal_actions(),
            'done': is_episode_done(self.tasks_done, self.num_tasks),
        }