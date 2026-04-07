import json
import math
import os
import torch
import lightning as L
from torch.utils.data import IterableDataset, DataLoader

from neutral_atoms.tasks import get_relevant_atoms
from neutral_atoms.moves import count_groups
from neutral_atoms.tasks import gates_to_moves
from neutral_atoms.map_generator import generate_random_map
from neutral_atoms.config import MAPS, _resolve_map, atom_map_to_positions

from .soft_cost import soft_layer_cost, hard_layer_cost
from .network import PlanNet


# ---- Dataset ----

class BoardDataset(IterableDataset):
    """Infinite stream of board instances. Each sample is a dict with atom_positions
    and tasks, either from a fixed map or randomly generated."""
    def __init__(self, cfg, map_data):
        self.cfg = cfg
        self.map_data = map_data

    def __iter__(self):
        while True:
            if self.cfg.random_board:
                m = generate_random_map(
                    (self.cfg.board_height, self.cfg.board_width),
                    self.cfg.num_qubits, self.cfg.num_tasks,
                    gates_per_layer=self.cfg.gates_per_layer,
                )
            else:
                m = self.map_data
            positions = atom_map_to_positions(m['atom_map'], self.cfg.board_width)
            atom_pos = torch.tensor(positions, dtype=torch.long)  # (Q, 2)
            # Flatten tasks to tensor-friendly form: (num_layers, gates_per_layer, 2)
            max_gates = max(len(layer) for layer in m['tasks'])
            tasks_tensor = torch.full((len(m['tasks']), max_gates, 2), -1, dtype=torch.long)
            for li, layer in enumerate(m['tasks']):
                for gi, (q1, q2) in enumerate(layer):
                    tasks_tensor[li, gi] = torch.tensor([q1, q2])
            yield atom_pos, tasks_tensor


def collate_boards(batch):
    atom_pos = torch.stack([b[0] for b in batch])    # (B, Q, 2)
    tasks = torch.stack([b[1] for b in batch])        # (B, L, G, 2)
    return atom_pos, tasks


# ---- Lightning Module ----

class IRWrapper(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = PlanNet(cfg)
        self.save_hyperparameters(cfg.to_dict())

    def _encode_layer_inputs(self, atom_positions, tasks_tensor, layer_idx):
        """Build per-qubit inputs for one gate layer.
        Args:
            atom_positions: (B, Q, 2) current qubit positions
            tasks_tensor: (B, L, G, 2) gate pairs (-1 padded)
            layer_idx: int
        Returns:
            src_cells: (B, Q) flat cell index
            gate_partner_cells: (B, Q) flat cell of gate partner (-1 if none)
            layer_idx_t: (B,) int tensor
            relevant_mask: (B, Q) bool — True for qubits involved in this layer's gates
            gate_pairs_list: list of B lists of (q1, q2) tuples
        """
        B, Q = atom_positions.shape[:2]
        W = self.cfg.board_width
        src_cells = atom_positions[:, :, 0] * W + atom_positions[:, :, 1]  # (B, Q)
        gate_partner_cells = torch.full((B, Q), -1, dtype=torch.long, device=self.device)
        relevant_mask = torch.zeros(B, Q, dtype=torch.bool, device=self.device)
        gate_pairs_list = []
        for b in range(B):
            pairs = []
            for gi in range(tasks_tensor.shape[2]):
                q1, q2 = tasks_tensor[b, layer_idx, gi].tolist()
                if q1 < 0:
                    break
                pairs.append((q1, q2))
                gate_partner_cells[b, q1] = src_cells[b, q2]
                gate_partner_cells[b, q2] = src_cells[b, q1]
                relevant_mask[b, q1] = True
                relevant_mask[b, q2] = True
            gate_pairs_list.append(pairs)
        layer_idx_t = torch.full((B,), layer_idx, dtype=torch.long, device=self.device)
        return src_cells, gate_partner_cells, layer_idx_t, relevant_mask, gate_pairs_list

    def _plan_one_layer(self, atom_positions, tasks_tensor, layer_idx):
        """Run refinement loop for one layer. Returns (logits, relevant_mask, gate_pairs_list)."""
        B, Q = atom_positions.shape[:2]
        src_cells, gate_partner_cells, layer_idx_t, relevant_mask, gate_pairs_list = \
            self._encode_layer_inputs(atom_positions, tasks_tensor, layer_idx)
        T = self.cfg.model.T
        plan_probs = None
        total_cost = atom_positions.new_tensor(0.0, dtype=torch.float)
        for t in range(T):
            # Select qubits to refine (all on first and last pass, random subset otherwise)
            if T == 1 or t == 0:
                selected = relevant_mask
            else:
                rand_select = torch.rand(B, Q, device=self.device) < self.cfg.model.select_prob
                selected = relevant_mask & rand_select
                selected = selected | (relevant_mask & ~selected.any(dim=-1, keepdim=True))  # at least 1
            logits = self.model(src_cells, gate_partner_cells, layer_idx_t,
                                plan_probs=plan_probs, selected=selected)
            # Mask non-relevant qubits: force them to stay at current cell
            stay_logits = torch.full_like(logits, -1e9)
            stay_logits.scatter_(2, src_cells.unsqueeze(-1), 0.0)
            logits = torch.where(relevant_mask.unsqueeze(-1), logits, stay_logits)
            # Compute soft cost for this refinement step (sum across batch)
            step_cost = self._batch_soft_cost(atom_positions, logits, relevant_mask, gate_pairs_list)
            total_cost = total_cost + step_cost
            # Prepare plan draft for next refinement step (detached)
            plan_probs = logits.detach().softmax(dim=-1)
        return total_cost / T, logits, relevant_mask, gate_pairs_list

    def _batch_soft_cost(self, atom_positions, logits, relevant_mask, gate_pairs_list):
        """Compute mean soft_layer_cost across the batch."""
        B = logits.shape[0]
        W = self.cfg.board_width
        K = self.cfg.training.top_k
        cw = self.cfg.training.collision_weight
        total = logits.new_tensor(0.0)
        for b in range(B):
            rel_idx = relevant_mask[b].nonzero(as_tuple=True)[0].tolist()
            if len(rel_idx) == 0:
                continue
            total = total + soft_layer_cost(
                atom_positions[b], logits[b, rel_idx], rel_idx,
                gate_pairs_list[b], W, top_k=K, collision_weight=cw,
                reconfig_weight=self.cfg.training.reconfig_weight,
            )
        return total / B

    def _apply_argmax_placement(self, atom_positions, logits, relevant_mask):
        """Apply argmax placement to get new atom_positions (detached, for next layer)."""
        B, Q = atom_positions.shape[:2]
        W = self.cfg.board_width
        dests = logits.argmax(dim=-1)  # (B, Q)
        dst_rows = dests // W
        dst_cols = dests % W
        new_pos = atom_positions.clone()
        new_pos[:, :, 0] = torch.where(relevant_mask, dst_rows, atom_positions[:, :, 0])
        new_pos[:, :, 1] = torch.where(relevant_mask, dst_cols, atom_positions[:, :, 1])
        return new_pos

    def training_step(self, batch, batch_idx):
        atom_positions, tasks_tensor = batch  # (B, Q, 2), (B, L, G, 2)
        atom_positions = atom_positions.to(self.device)
        tasks_tensor = tasks_tensor.to(self.device)
        num_layers = tasks_tensor.shape[1]
        total_loss = atom_positions.new_tensor(0.0, dtype=torch.float)
        pos = atom_positions
        for layer_idx in range(num_layers):
            layer_cost, logits, rel_mask, _ = self._plan_one_layer(pos, tasks_tensor, layer_idx)
            total_loss = total_loss + layer_cost
            pos = self._apply_argmax_placement(pos, logits.detach(), rel_mask)
        loss = total_loss / num_layers
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        atom_positions, tasks_tensor = batch
        atom_positions = atom_positions.to(self.device)
        tasks_tensor = tasks_tensor.to(self.device)
        B = atom_positions.shape[0]
        num_layers = tasks_tensor.shape[1]
        W = self.cfg.board_width
        total_hard = 0.0
        pos = atom_positions
        for layer_idx in range(num_layers):
            _, logits, rel_mask, gate_pairs_list = self._plan_one_layer(pos, tasks_tensor, layer_idx)
            for b in range(B):
                rel_idx = rel_mask[b].nonzero(as_tuple=True)[0].tolist()
                if len(rel_idx) > 0:
                    total_hard += hard_layer_cost(
                        pos[b], logits[b, rel_idx], rel_idx, gate_pairs_list[b], W,
                    )
            pos = self._apply_argmax_placement(pos, logits.detach(), rel_mask)
        avg_cost = total_hard / B
        self.log('val_hard_cost', avg_cost, prog_bar=True)
        return avg_cost

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        warmup = self.cfg.training.warmup_steps
        total = self.cfg.training.max_steps
        def lr_fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def train_dataloader(self):
        m = _resolve_map(MAPS[self.cfg.map_num])
        ds = BoardDataset(self.cfg, m)
        return DataLoader(ds, batch_size=self.cfg.training.batch_size,
                          collate_fn=collate_boards, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        m = _resolve_map(MAPS[self.cfg.map_num])
        ds = BoardDataset(self.cfg, m)
        return DataLoader(ds, batch_size=self.cfg.training.batch_size,
                          collate_fn=collate_boards, num_workers=0)

    @torch.no_grad()
    def infer(self, map_data):
        """Run inference on a single map instance. Returns (solution_dict, total_cost).
        solution_dict is atom-viz compatible JSON format."""
        self.eval()
        W = map_data['board_dim'][1]
        H = map_data['board_dim'][0]
        positions = atom_map_to_positions(map_data['atom_map'], W)
        atom_pos = torch.tensor(positions, dtype=torch.long, device=self.device)  # (Q, 2)
        tasks = map_data['tasks']
        # Build tasks_tensor: (1, L, G, 2)
        max_gates = max(len(layer) for layer in tasks)
        tasks_tensor = torch.full((1, len(tasks), max_gates, 2), -1,
                                  dtype=torch.long, device=self.device)
        for li, layer in enumerate(tasks):
            for gi, (q1, q2) in enumerate(layer):
                tasks_tensor[0, li, gi] = torch.tensor([q1, q2])
        pos = atom_pos.unsqueeze(0)  # (1, Q, 2)
        initial_atoms = {str(i): {'row': r, 'col': c} for i, (r, c) in enumerate(positions)}
        plan = []
        total_cost = 0
        for layer_idx in range(len(tasks)):
            _, logits, rel_mask, gate_pairs_list = self._plan_one_layer(pos, tasks_tensor, layer_idx)
            # Extract moves for this layer
            dests = logits[0].argmax(dim=-1)  # (Q,)
            layer_moves = []
            reconfig_move_tensors = []
            for q in range(atom_pos.shape[0]):
                if not rel_mask[0, q]:
                    continue
                src_r, src_c = pos[0, q].tolist()
                dst_flat = dests[q].item()
                dst_r, dst_c = dst_flat // W, dst_flat % W
                if src_r != dst_r or src_c != dst_c:
                    layer_moves.append({
                        'atom': q,
                        'from': {'row': src_r, 'col': src_c},
                        'to': {'row': dst_r, 'col': dst_c},
                    })
                    reconfig_move_tensors.append(
                        torch.tensor([src_r, src_c, dst_r, dst_c], dtype=torch.long))
            plan.append(layer_moves)
            # Compute actual cost for this layer
            if reconfig_move_tensors:
                total_cost += count_groups(torch.stack(reconfig_move_tensors), canonicalize=False)
            # Apply placement for next layer
            pos = self._apply_argmax_placement(pos, logits.detach(), rel_mask)
            # Gate cost from final positions
            gate_moves = gates_to_moves(tasks[layer_idx],
                                        pos[0].cpu())
            if len(gate_moves) > 0:
                total_cost += 2 * count_groups(gate_moves, canonicalize=True)
        solution = {
            'board': {'rows': H, 'cols': W, 'initialAtoms': initial_atoms},
            'circuit': tasks,
            'plan': plan,
        }
        return solution, total_cost

    def save_solution(self, map_data, path):
        """Run inference and save atom-viz compatible JSON."""
        solution, cost = self.infer(map_data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(solution, f, indent=2)
        print(f"Solution saved to {path} (cost={cost})")
        return solution, cost
