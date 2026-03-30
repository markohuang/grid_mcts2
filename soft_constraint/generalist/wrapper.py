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
from surrogate.primitives import true_total_cost


def neutral_atoms_constraint_loss(probs, init_cells, tasks, task_partner, H, W, N,
                                   lambda_g, lambda_r, lambda_aux, loss_mode, delta):
    """Cost surrogate loss with relevant-atom pinning.

    Non-relevant atoms (no gate in a layer) are pinned to their previous-layer
    position. Only relevant atoms' model predictions contribute to the loss.
    """
    B, TN, C = probs.shape
    T = len(tasks[0])
    plan = probs.view(B, T, N, C)
    relevant_mask = (task_partner.view(B, T, N) != N)  # True if atom has gate

    total_loss = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
    for b in range(B):
        init_dists_b = F.one_hot(init_cells[b], C).float().to(probs.dtype)
        mixed_layers = []
        prev = init_dists_b
        for t in range(T):
            pred = plan[b, t]                              # (N, C)
            rel = relevant_mask[b, t].unsqueeze(-1)        # (N, 1)
            mixed = torch.where(rel, pred, prev.detach())  # pin non-relevant
            mixed_layers.append(mixed)
            prev = mixed
        cost, _ = total_cost_surrogate(
            init_dists_b, mixed_layers, tasks[b], H, W,
            lambda_g=lambda_g, lambda_r=lambda_r, lambda_aux=lambda_aux,
            loss_mode=loss_mode, delta=delta)
        total_loss = total_loss + cost
    return total_loss / B


def _na_soft_init(B, ans_len, C, device):
    """Uniform 1/C initialization for all answer tokens."""
    return torch.full((B, ans_len, C), 1.0 / C, device=device)


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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        init_cells, task_partner, tasks_list = batch
        B = init_cells.shape[0]
        tp_flat = task_partner.view(B, -1)  # (B, T*N)
        x_ans_soft = _na_soft_init(B, self.ans_len, self.C, init_cells.device)

        T_outer = self.cfg.model.T
        z_H, z_L, total_loss = None, None, 0.0
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for step in range(T_outer):
                S = (torch.rand(B, self.ans_len, device=init_cells.device) < self.cfg.training.select_prob) \
                    if T_outer > 1 else torch.ones(B, self.ans_len, device=init_cells.device, dtype=torch.bool)
                logits, z_H, z_L = self.model(
                    init_cells, x_ans_soft, tp_flat, z_H, z_L, selected=S)

                probs = logits.float().softmax(-1)  # float32 for surrogate stability
                total_loss = total_loss + neutral_atoms_constraint_loss(
                    probs, init_cells, tasks_list, task_partner,
                    self.H, self.W, self.N,
                    self.cfg.surrogate.lambda_g, self.cfg.surrogate.lambda_r,
                    self.cfg.surrogate.lambda_aux, self.cfg.surrogate.loss_mode,
                    self.cfg.surrogate.delta)

        loss = total_loss / T_outer
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)

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
        x_ans_soft = _na_soft_init(B, self.ans_len, self.C, init_cells.device)
        S = torch.ones(B, self.ans_len, device=init_cells.device, dtype=torch.bool)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits, _, _ = self.model(init_cells, x_ans_soft, tp_flat, selected=S)
            probs = logits.float().softmax(-1)
            val_loss = neutral_atoms_constraint_loss(
                probs, init_cells, tasks_list, task_partner,
                self.H, self.W, self.N,
                self.cfg.surrogate.lambda_g, self.cfg.surrogate.lambda_r,
                self.cfg.surrogate.lambda_aux, self.cfg.surrogate.loss_mode,
                self.cfg.surrogate.delta)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Compute true cost on hard assignments
        hard_plan = logits.argmax(-1).view(B, self.T_layers, self.N)   # (B, T, N)
        total_true_cost = 0.0
        for b in range(B):
            init_pos = init_cells[b].cpu().numpy()
            plan_pos = [hard_plan[b, t].cpu().numpy() for t in range(self.T_layers)]
            tc, _ = true_total_cost(init_pos, plan_pos, tasks_list[b], self.H, self.W)
            total_true_cost += tc
        avg_true_cost = total_true_cost / B
        self.log('val_true_cost', avg_true_cost, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def infer(self, init_cells, task_partner, n_steps=1):
        B = init_cells.shape[0]
        tp_flat = task_partner.view(B, -1)
        x_ans_soft = _na_soft_init(B, self.ans_len, self.C, init_cells.device)
        z_H, z_L = None, None
        for step in range(n_steps):
            S = (torch.rand(B, self.ans_len, device=init_cells.device) < self.cfg.training.select_prob) \
                if step < n_steps - 1 else torch.ones(B, self.ans_len, device=init_cells.device, dtype=torch.bool)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, z_H, z_L = self.model(
                    init_cells, x_ans_soft, tp_flat, z_H, z_L, selected=S)
        hard_plan = logits.argmax(-1).view(B, self.T_layers, self.N)
        return hard_plan

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
