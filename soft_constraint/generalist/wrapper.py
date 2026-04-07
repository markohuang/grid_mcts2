"""Lightning wrapper for neutral atoms IR training."""
import math
import torch
from torch.nn import functional as F
from lightning.pytorch import LightningModule

from generalist.models import get_backbone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surrogate.feasibility import total_cost_surrogate
from surrogate.batched import batched_total_cost_surrogate
from surrogate.primitives import true_total_cost


def neutral_atoms_constraint_loss(plan_dists, init_cells, tasks, task_partner, H, W, N,
                                   lambda_g, lambda_r, lambda_aux, loss_mode, delta):
    """Cost surrogate loss with relevant-atom pinning.

    Args:
        plan_dists: (B, T*N, C) full plan distributions (draft with updates merged)
    """
    B, TN, C = plan_dists.shape
    T = len(tasks[0])
    plan = plan_dists.view(B, T, N, C)
    relevant_mask = (task_partner.view(B, T, N) != N)

    init_dists = F.one_hot(init_cells, C).float().to(plan_dists.dtype)

    # Pin non-relevant atoms to previous layer
    mixed_layers = []
    prev = init_dists
    for t in range(T):
        pred = plan[:, t]
        rel = relevant_mask[:, t].unsqueeze(-1)
        mixed = torch.where(rel, pred, prev.detach())
        mixed_layers.append(mixed)
        prev = mixed

    shared_tasks = _all_tasks_equal(tasks)
    if shared_tasks:
        costs = batched_total_cost_surrogate(
            init_dists, mixed_layers, tasks[0], H, W,
            lambda_g=lambda_g, lambda_r=lambda_r, loss_mode=loss_mode, delta=delta)
        return costs.mean()

    total_loss = torch.tensor(0.0, device=plan_dists.device, dtype=plan_dists.dtype)
    for b in range(B):
        layers_b = [mixed_layers[t][b] for t in range(T)]
        cost, _ = total_cost_surrogate(
            init_dists[b], layers_b, tasks[b], H, W,
            lambda_g=lambda_g, lambda_r=lambda_r, lambda_aux=lambda_aux,
            loss_mode=loss_mode, delta=delta)
        total_loss = total_loss + cost
    return total_loss / B


def _all_tasks_equal(tasks):
    ref = tasks[0]
    return all(t == ref for t in tasks[1:])


def _draft_init_biased(init_cells, N, T_layers, C, device, bias=4.0):
    """Initialize draft biased toward initial positions."""
    B = init_cells.shape[0]
    logits = torch.zeros(B, T_layers * N, C, device=device)
    # Repeat init_cells across all layers
    init_expanded = init_cells.unsqueeze(1).expand(B, T_layers, N).reshape(B, T_layers * N)
    logits.scatter_(2, init_expanded.unsqueeze(-1), bias)
    return logits.softmax(-1)


class NeutralAtomsWrapper(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = get_backbone(cfg.model_type, cfg.model)
        self._use_muon = (cfg.optim.type == 'muon')
        if self._use_muon:
            self.automatic_optimization = False
        self._accum_steps = cfg.accumulate_grad_batches

        self.H = cfg.env.board_height
        self.W = cfg.env.board_width
        self.N = cfg.env.num_qubits
        self.T_layers = cfg.env.num_layers
        self.C = self.H * self.W
        self.ans_len = self.T_layers * self.N

    def _relevant_mask(self, task_partner, B):
        """(B, T*N) bool — True for atoms involved in a gate for that layer."""
        return (task_partner.view(B, -1) != self.N)

    def _compute_loss(self, plan_dists, init_cells, tasks_list, task_partner):
        return neutral_atoms_constraint_loss(
            plan_dists, init_cells, tasks_list, task_partner,
            self.H, self.W, self.N,
            self.cfg.surrogate.lambda_g, self.cfg.surrogate.lambda_r,
            self.cfg.surrogate.lambda_aux, self.cfg.surrogate.loss_mode,
            self.cfg.surrogate.delta)

    # ------------------------------------------------------------------
    # Training — iterative partial refinement
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        init_cells, task_partner, tasks_list = batch
        B = init_cells.shape[0]
        tp_flat = task_partner.view(B, -1)
        relevant = self._relevant_mask(task_partner, B)  # (B, T*N)
        select_prob = self.cfg.training.select_prob

        T_outer = self.cfg.model.T
        total_loss = 0.0

        # Start from biased draft (near initial positions)
        x_draft = _draft_init_biased(init_cells, self.N, self.T_layers, self.C,
                                     init_cells.device)
        z_H, z_L = None, None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for step in range(T_outer):
                # Random subset of positions to update this step
                S = (torch.rand(B, self.ans_len, device=init_cells.device) < select_prob) & relevant

                logits, z_H, z_L = self.model(
                    init_cells, x_draft, tp_flat, z_H, z_L, selected=S)

                probs = logits.float().softmax(-1)

                # Partial update: only selected positions get model's prediction,
                # rest keep draft values
                updated_draft = torch.where(
                    S.unsqueeze(-1), probs, x_draft.detach().to(probs.dtype))

                total_loss = total_loss + self._compute_loss(
                    updated_draft, init_cells, tasks_list, task_partner)

                # Evolve draft for next step (detach to prevent BPTT across steps)
                x_draft = updated_draft.detach()

        loss = total_loss / T_outer
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=B)

        if self._use_muon:
            self._muon_opt_step(loss, batch_idx)
        else:
            return loss

    def _muon_opt_step(self, loss, batch_idx):
        adamw, muon = self.optimizers()
        self.manual_backward(loss / self._accum_steps)
        if (batch_idx + 1) % self._accum_steps == 0:
            adamw.step(); muon.optimizer.step()
            adamw.zero_grad(); muon.zero_grad()
            for sched in self.lr_schedulers():
                sched.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        init_cells, task_partner, tasks_list = batch
        B = init_cells.shape[0]
        tp_flat = task_partner.view(B, -1)
        relevant = self._relevant_mask(task_partner, B)

        # Inference with multiple refinement steps
        x_draft = _draft_init_biased(init_cells, self.N, self.T_layers, self.C,
                                     init_cells.device)

        T_outer = self.cfg.model.T
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            z_H, z_L = None, None
            for step in range(T_outer):
                # Select all relevant atoms for val (no randomness)
                S = relevant
                logits, z_H, z_L = self.model(
                    init_cells, x_draft, tp_flat, z_H, z_L, selected=S)
                probs = logits.float().softmax(-1)
                x_draft = torch.where(
                    S.unsqueeze(-1), probs, x_draft.to(probs.dtype)).detach()

            val_loss = self._compute_loss(x_draft, init_cells, tasks_list, task_partner)

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B)

        # True cost on hard assignments
        hard_plan = x_draft.argmax(-1).view(B, self.T_layers, self.N)
        total_true_cost = 0.0
        for b in range(B):
            tc, _ = true_total_cost(
                init_cells[b].cpu().numpy(),
                [hard_plan[b, t].cpu().numpy() for t in range(self.T_layers)],
                tasks_list[b], self.H, self.W)
            total_true_cost += tc
        self.log('val_true_cost', total_true_cost / B, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=B)

    def infer(self, init_cells, task_partner, n_steps=None):
        B = init_cells.shape[0]
        tp_flat = task_partner.view(B, -1)
        relevant = self._relevant_mask(task_partner, B)
        if n_steps is None:
            n_steps = self.cfg.model.T

        x_draft = _draft_init_biased(init_cells, self.N, self.T_layers, self.C,
                                     init_cells.device)
        z_H, z_L = None, None
        for step in range(n_steps):
            S = relevant
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, z_H, z_L = self.model(
                    init_cells, x_draft, tp_flat, z_H, z_L, selected=S)
            probs = logits.float().softmax(-1)
            x_draft = torch.where(
                S.unsqueeze(-1), probs, x_draft.to(probs.dtype)).detach()
        return x_draft.argmax(-1).view(B, self.T_layers, self.N)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        warmup = self.cfg.optim.warmup_steps
        total = self.cfg.trainer.max_steps

        def lr_fn(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        if not self._use_muon:
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.optim.lr,
                                    weight_decay=self.cfg.optim.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

        from generalist.muon import Muon
        muon_params, adamw_params = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (muon_params if p.ndim == 2 else adamw_params).append(p)

        adamw = torch.optim.AdamW(adamw_params, lr=self.cfg.optim.lr,
                                  weight_decay=self.cfg.optim.weight_decay)
        muon = Muon(muon_params, lr=self.cfg.optim.lr, momentum=self.cfg.optim.muon_momentum,
                    weight_decay=self.cfg.optim.weight_decay, nesterov=True,
                    adjust_lr_fn='match_rms_adamw')

        return (
            [adamw, muon],
            [
                {'scheduler': torch.optim.lr_scheduler.LambdaLR(adamw, lr_fn), 'interval': 'step'},
                {'scheduler': torch.optim.lr_scheduler.LambdaLR(muon, lr_fn), 'interval': 'step'},
            ],
        )
