# Iterative Refinement for Sudoku: Architecture Overview

directory location for reference: `/home/marko/uai_sudoku_comp`

## Philosophy

The model solves sudoku by **iterative soft refinement** — not by classifying each cell independently, but by maintaining a soft probability board and repeatedly refining it through dual-stream reasoning passes. Supervision comes from **constraint satisfaction** (rows/cols/boxes must contain each digit exactly once) rather than per-cell cross-entropy against ground truth.

Key ideas:
- **Dual-stream processing**: a "draft" stream (z_H) holds the answer, a "scratchpad" stream (z_L) does internal reasoning — only z_H is read out
- **Soft board state**: cells hold probability distributions, not hard assignments — this lets gradients flow through the full refinement chain
- **Constraint loss**: MSE between row/col/box digit-sums and the target (all ones) — the model discovers valid digit placement patterns rather than memorizing solutions
- **ACT (Adaptive Computation Time) passes**: the wrapper loops T times, each time selecting a random subset of cells and passing the carried z_H/z_L state forward
- **Weight sharing**: the same small network (`l_level`) is called `H_cycle * (n+1)` times per forward pass — depth comes from iteration, not parameter count

## File Structure

```
uai_sudoku_comp/
├── main.py                           # CLI entry point (absl + ml_collections)
├── config.py                         # ml_collections bridge
├── wrapper.py                        # SudokuWrapper (Lightning module)
├── muon.py                           # Muon optimizer
│
├── configs/
│   ├── __init__.py                   # get_config("experiment/model/variant")
│   ├── base.py                       # universal defaults (optim, trainer, etc.)
│   ├── model_configs/
│   │   ├── __init__.py               # MODEL_CONFIGS registry
│   │   ├── trm_dit_config.py         # transformer variants (29 entries)
│   │   └── trm_unet_config.py        # conv variants (21 entries)
│   └── experiments/
│       ├── __init__.py               # EXPERIMENTS registry
│       └── sudoku.py                 # vocab, grid, data, augmentation defaults
│
├── models/
│   ├── __init__.py                   # get_backbone(model_type, cfg) registry
│   ├── ir_backbone.py                # IRBackbone (dual-stream core)
│   ├── trm_blocks.py                 # TransformerBlock, ResidualBlock, GridEmbedding, etc.
│   ├── trm_dit.py                    # DiTModel (transformer path)
│   └── trm_unet.py                   # UNetModel (convolutional path)
│
├── experiments/
│   └── sudoku/
│       ├── __init__.py               # setup_experiment(cfg) → (tloader, vloader)
│       ├── dataset.py                # SudokuStreamDataset (infinite iterator)
│       └── augmentations.py          # digit perm, rotation, reflection, transpose
│
├── data/sudoku/                      # pre-generated .pt files
│   ├── sudoku_39_50_{train,test}.pt  # puzzles (one-hot)
│   ├── labels_39_50_{train,test}.pt  # solutions (one-hot)
│   ├── sudoku_47_64_{train,test}.pt
│   └── labels_47_64_{train,test}.pt
│
├── eval_ensemble.py                  # ensemble inference (rowcol + box models)
├── callbacks.py                      # visualization, registry, metrics
└── visualization.py                  # sudoku grid plotting
```

## Config System

4-layer composition, all via `ml_collections.ConfigDict`:

```
Layer 1: base.py          → universal defaults (lr, warmup, trainer flags)
Layer 2: experiments/      → experiment-specific (vocab_size, grid_size, data paths, augmentations)
Layer 3: model_configs/    → architecture hyperparams (hidden_size, num_blocks, T, H_cycle, n)
Layer 4: cross-injection   → experiment tells model shared dims (vocab_size, grid_size → model.*)
```

### CLI usage

```bash
# Format: config.py:<experiment>/<model_type>/<variant>
python main.py --cfg=config.py:sudoku/trm_dit/iter_T4N4
python main.py --cfg=config.py:sudoku/trm_unet/iter_T2H2N8

# Shorthand (defaults to sudoku experiment)
python main.py --cfg=config.py:trm_dit/iter_T4N4

# Override any config field
python main.py --cfg=config.py:trm_dit/iter_T4N4 \
  --cfg.optim.lr=1e-4 \
  --cfg.trainer.max_steps=100000 \
  --cfg.training.constraint_loss=rowcol
```

### Config string parsing

```python
# configs/__init__.py
def _parse_config_string(config_string):
    parts = config_string.split('/')
    # 3 parts: "sudoku/trm_dit/iter_T4N4" → explicit
    # 2 parts: "trm_dit/iter_T4N4"        → default experiment
    #          "sudoku/trm_dit"            → variant='base'
    # 1 part:  "trm_dit"                  → default experiment + variant='base'
```

### Variant naming convention

`<tier>_T<T>H<H_cycle>N<n>` — e.g. `iter_T4H2N4` means:
- **tier**: param budget (`base` ~2.3M, `iter` ~1M)
- **T**: ACT supervision passes (wrapper loop count)
- **H_cycle**: outer IR cycles (gradient truncation on all but last)
- **n**: inner z_L refinement updates per cycle

Effective depth per forward = `H_cycle * (n + 1) * num_blocks`

### Example variant definitions

```python
# configs/model_configs/trm_dit_config.py
VARIANTS = {
    'base_T4N2':  dict(T=4, n=2, num_blocks=3, hidden_size=192, num_heads=4, ctx_dim=192),
    'iter_T4N4':  dict(T=4, n=4, num_blocks=2, hidden_size=144, num_heads=4, ctx_dim=144),
    'iter_T2H2N8': dict(T=2, H_cycle=2, n=8, num_blocks=2, hidden_size=144, num_heads=4, ctx_dim=144),
}

# configs/model_configs/trm_unet_config.py
VARIANTS = {
    'base_T4N2':    dict(T=4, n=2, num_blocks=4, model_ch=160, ctx_dim=256),
    'iter_T4N2':    dict(T=4, n=2, num_blocks=2, model_ch=128, ctx_dim=256),
    'iter_T2H2N8':  dict(T=2, H_cycle=2, n=8, num_blocks=2, model_ch=128, ctx_dim=256),
}
```

## Data Pipeline

### Storage format
- Pre-generated `.pt` files: one-hot `(N, 9, 9, 9)` → converted to integer `(N, 9, 9)` on load
- `0` = empty cell, `1-9` = digits
- Train difficulty `39_50` (39-50 given cells), val difficulty `47_64` (47-64 given cells)

### Loading

```python
# experiments/sudoku/__init__.py
def onehot_to_int(onehot):
    digits = onehot.argmax(-1) + 1       # 1-9
    empty = onehot.sum(-1) == 0          # all-zero = empty
    digits[empty] = 0
    return digits.long()

def setup_experiment(cfg):
    # Train: concat train+test splits of train difficulty
    p_train, l_train = _load_split('39_50', 'train')
    p_test, l_test   = _load_split('39_50', 'test')
    puzzles_train = torch.cat([p_train, p_test])

    # Val: test split of val difficulty
    puzzles_val, labels_val = _load_split('47_64', 'test')
```

### Streaming dataset

```python
# experiments/sudoku/dataset.py
class SudokuStreamDataset(IterableDataset):
    def __iter__(self):
        while True:
            idx = random.randrange(self.n)
            q_in, q_out = self.puzzles[idx], self.solutions[idx]
            if self.transform is not None:
                ep = self.transform({'q_in': q_in, 'q_out': q_out})
                q_in, q_out = ep['q_in'], ep['q_out']
            yield q_in, q_out  # (9, 9), (9, 9)
```

### Augmentations

Applied per-sample at load time. All are constraint-preserving:

```python
# experiments/sudoku/augmentations.py
def apply_transforms(episode, transforms):
    for t in transforms:
        if t == 'digit_permutation':
            perm = torch.zeros(10, dtype=torch.long)
            perm[1:] = torch.randperm(9) + 1       # random bijection on 1-9, 0→0
            q_in, q_out = perm[q_in], perm[q_out]
        elif t == 'rotation':
            k = torch.randint(0, 4, (1,)).item()    # 0/90/180/270 degrees
            q_in = torch.rot90(q_in, k, (-2, -1))
        elif t == 'reflection':
            if torch.rand(1) > 0.5:                 # horizontal flip
                q_in = torch.flip(q_in, [-1])
        elif t == 'transpose':
            if torch.rand(1) > 0.5:
                q_in = q_in.transpose(-2, -1)
```

## IRBackbone: The Core Engine

```
                    ┌─────────────────────────────────────────┐
                    │            IRBackbone                    │
                    │                                         │
                    │  for cycle in range(H_cycle):           │
                    │    ┌───────────────────────────────┐    │
                    │    │  for _ in range(n):            │    │
                    │    │    z_L = l_level(z_L,          │    │
                    │    │              z_H + injection)  │    │
                    │    │                                │    │
                    │    │  z_H = l_level(z_H, z_L)      │    │
                    │    └───────────────────────────────┘    │
                    │                                         │
                    │  gradient: no_grad on first H_cycle-1   │
                    │            grad on last cycle only       │
                    └─────────────────────────────────────────┘
```

- **z_H** (answer draft): holds the current best guess. Read out at the end for logits
- **z_L** (scratchpad): internal reasoning state. Sees z_H + injection but is never read out
- **injection**: focus signal — tells the model which cells are currently selected for update
- **l_level**: the shared block stack (TransformerLLevel or UNetLLevel). Called `n` times for z_L, then once for z_H

```python
# models/ir_backbone.py
class IRBackbone(nn.Module):
    def __init__(self, l_level, H_cycle=1, n=1):
        self.l_level = l_level
        self.H_cycle = H_cycle
        self.n = n

    def forward(self, z_H, z_L, injection, **l_level_kwargs):
        def h_cycle(z_H, z_L):
            for _ in range(self.n):
                z_L = self.l_level(z_L, z_H + injection, **l_level_kwargs)
            z_H = self.l_level(z_H, z_L, **l_level_kwargs)
            return z_H, z_L

        with torch.no_grad():
            for _ in range(self.H_cycle - 1):
                z_H, z_L = h_cycle(z_H, z_L)
        return h_cycle(z_H, z_L)
```

## DiTModel (Transformer Path)

- z_H shape: `(B, 2*81, hidden_size)` — first 81 = question embedding, last 81 = answer embedding
- z_L shape: same, initialized to zeros
- injection: learnable `selected_emb` vector placed at selected positions in the answer half
- l_level: `TransformerLLevel` — stack of `TransformerBlock`s (self-attn → SwiGLU → residual)

```python
# models/trm_dit.py
class DiTModel(nn.Module):
    def __init__(self, cfg):
        self.grid_emb = GridEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.grid_size)
        l_level = TransformerLLevel(cfg.hidden_size, cfg.num_heads, cfg.num_blocks,
                                    0, cfg.expansion, cfg.dropout, False)
        self.ir = IRBackbone(l_level, cfg.H_cycle, cfg.n)
        self.selected_emb = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.out_norm = nn.LayerNorm(cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size + 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q_in, x_ans_soft, z_H=None, z_L=None, selected=None):
        if z_H is None:
            q_emb   = self.grid_emb(q_in, 0)                              # (B, 81, d)
            ans_emb = self.grid_emb.forward_soft(x_ans_soft, self.seq_len) # (B, 81, d)
            z_H = torch.cat([q_emb, ans_emb], dim=1)                      # (B, 162, d)
        z_L = z_L if z_L is not None else torch.zeros_like(z_H)

        injection = torch.zeros_like(z_H)
        injection[:, self.seq_len:] = selected.unsqueeze(-1).float() * self.selected_emb

        z_H, z_L = self.ir(z_H, z_L, injection)
        logits = self.out_proj(self.out_norm(z_H[:, self.seq_len:]))       # (B, 81, V+1)
        return logits, z_H.detach(), z_L.detach()
```

### GridEmbedding: hard vs soft input

```python
# models/trm_blocks.py
class GridEmbedding(nn.Module):
    def forward(self, tokens, pos_offset=0):
        tok_emb = self.embed_tokens(tokens.flatten(1).clamp(0, self.vocab_size))
        return self.scale * (tok_emb + self.pos_embed[pos_offset:pos_offset + self.seq_len])

    def forward_soft(self, probs, pos_offset=0):
        tok_emb = probs @ self.embed_tokens.weight  # (B,81,V+1) @ (V+1,d) → (B,81,d)
        return self.scale * (tok_emb + self.pos_embed[pos_offset:pos_offset + self.seq_len])
```

Key: `forward` takes integer tokens, `forward_soft` takes probability distributions and does a weighted sum over embedding vectors. This is what makes the soft board differentiable.

## UNetModel (Convolutional Path)

- z_H shape: `(B, model_ch, 9, 9)` — spatial feature map from conv over `[q_onehot | ans_onehot]`
- z_L shape: same, initialized to zeros
- injection: `inj_conv(selected.view(B, 1, 9, 9))` — 1-channel selection mask convolved to `model_ch`
- l_level: `UNetLLevel` — stack of `ResidualBlock`s (conv → GroupNorm → SiLU → conv → residual)

```python
# models/trm_unet.py
class UNetModel(nn.Module):
    def __init__(self, cfg):
        V, C = cfg.vocab_size, cfg.vocab_size + 1
        self.in_conv  = nn.Conv2d(2 * C, cfg.model_ch, 3, padding=1)
        self.inj_conv = nn.Conv2d(1, cfg.model_ch, 3, padding=1)
        l_level = UNetLLevel(cfg.model_ch, cfg.num_blocks, 0, False)
        self.ir = IRBackbone(l_level, cfg.H_cycle, cfg.n)
        self.out_norm = nn.GroupNorm(8, cfg.model_ch)
        self.out_conv = nn.Conv2d(cfg.model_ch, V + 1, 3, padding=1)

    def forward(self, q_in, x_ans_soft, z_H=None, z_L=None, selected=None):
        B, G = q_in.shape[0], self.grid_size
        if z_H is None:
            q_oh   = F.one_hot(q_in.long(), self.num_classes).permute(0,3,1,2).float()  # (B,C,9,9)
            ans_oh = x_ans_soft.view(B,G,G,self.num_classes).permute(0,3,1,2)           # (B,C,9,9)
            z_H = self.in_conv(torch.cat([q_oh, ans_oh], dim=1))                        # (B,ch,9,9)
        z_L = z_L if z_L is not None else torch.zeros_like(z_H)
        injection = self.inj_conv(selected.view(B, 1, G, G).float())                    # (B,ch,9,9)
        z_H, z_L = self.ir(z_H, z_L, injection)
        logits = self.out_conv(F.silu(self.out_norm(z_H))).permute(0,2,3,1).flatten(1,2) # (B,81,V+1)
        return logits, z_H.detach(), z_L.detach()
```

## DiT vs UNet: Side-by-Side

| Aspect | DiTModel | UNetModel |
|---|---|---|
| z_H shape | `(B, 162, d)` sequence | `(B, ch, 9, 9)` spatial |
| z_H init | `cat(grid_emb(q), grid_emb_soft(ans))` | `conv3x3(cat(q_onehot, ans_onehot))` |
| injection | learnable `selected_emb` at positions | `conv3x3(selection_mask)` |
| l_level | TransformerLLevel (self-attn + SwiGLU) | UNetLLevel (ResidualBlocks with conv) |
| readout | `linear(layernorm(z_H[:, 81:]))` | `conv3x3(silu(groupnorm(z_H)))` → flatten |
| position info | 2D sincos pos embed (baked into GridEmbedding) | implicit from spatial layout |
| shared IR core | identical IRBackbone | identical IRBackbone |

## Constraint Loss

Instead of cross-entropy per cell, the loss measures how well the predicted probability board satisfies sudoku rules.

```python
# wrapper.py
def sudoku_constraint_loss(probs, mode='all'):
    # probs: (B, 81, 9) — probabilities for digits 1-9 only (class 0 excluded)
    B, G2, V = probs.shape
    G = int(G2 ** 0.5)               # 9
    grid = probs.view(B, G, G, V)    # (B, 9, 9, 9) = (B, row, col, digit)
    target = torch.ones(B, G, G)     # each (row, digit) sum should be 1.0

    loss = 0.0
    if mode in ('all', 'rowcol'):
        loss += F.mse_loss(grid.sum(2), target)  # row constraint: sum over columns
        loss += F.mse_loss(grid.sum(1), target)  # col constraint: sum over rows
    if mode in ('all', 'box'):
        box = int(G ** 0.5)  # 3
        # Reshape into 3x3 boxes, sum cells within each box per digit
        p = grid.unfold(1, box, box).unfold(2, box, box).contiguous().view(B, G, G, V)
        loss += F.mse_loss(p.sum(-1), target)
    return loss
```

Constraint modes:
- `'all'`: row + col + box (default)
- `'rowcol'`: row + col only
- `'box'`: box only

This is what makes ensemble possible — train one model with `rowcol`, another with `box`, combine at inference.

## Soft Board Initialization

```python
# wrapper.py
def _soft_init(q_in, vocab_size):
    # q_in: (B, 9, 9) integer grid
    # Returns: (B, 81, 11) soft probability board
    #   given cells → one-hot at their digit
    #   free cells  → uniform 1/9 over digits 1-9, zero at index 0
    q_flat = q_in.view(B, G2)
    free_flat = (q_flat == 0)
    x_soft = F.one_hot(q_flat.long(), vocab_size + 1).float()
    uniform = torch.zeros(vocab_size + 1)
    uniform[1:vocab_size] = 1.0 / (vocab_size - 1)  # 1/9 each
    x_soft[free_flat] = uniform
    return x_soft
```

## Training Loop

```
┌──────────────────────────────────────────────────────────────────┐
│ training_step(batch)                                             │
│                                                                  │
│   q_in, q_out = batch                    # (B,9,9), (B,9,9)     │
│   x_ans_soft = _soft_init(q_in)          # (B,81,11) uniform    │
│   z_H, z_L = None, None                                         │
│                                                                  │
│   for step in range(T):        ◄── ACT supervision loop         │
│     S = random 30% of free cells         # (B,81) bool mask     │
│     logits, z_H, z_L = model(q_in, x_ans_soft, z_H, z_L, S)    │
│     probs = logits.softmax(-1)                                   │
│     mixed = where(given, onehot_given, probs)  # pin givens     │
│     total_loss += constraint_loss(mixed[:,:,1:10])               │
│                                                                  │
│   loss = total_loss / T                                          │
│   ► z_H, z_L detached between ACT steps (no BPTT across T)      │
│   ► gradients flow through the final H_cycle of IR only          │
└──────────────────────────────────────────────────────────────────┘
```

```python
# wrapper.py — training_step
def training_step(self, batch, batch_idx):
    q_in, q_out = batch
    B, G, G2 = q_in.shape[0], self.grid_size, self.grid_size ** 2
    free_flat    = (q_in == 0).view(B, G2)
    given_mask   = (~free_flat).unsqueeze(-1)
    onehot_given = F.one_hot(q_in.view(B, G2).long(), self.cfg.vocab_size + 1).float()
    x_ans_soft = _soft_init(q_in, self.cfg.vocab_size)

    T = self.cfg.model.T
    z_H, z_L, total_loss = None, None, 0.0
    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        for step in range(T):
            S = (torch.rand(B, G2, device=q_in.device) < self.cfg.training.select_prob) & free_flat \
                if T > 1 else free_flat
            logits, z_H, z_L = self.model(q_in, x_ans_soft, z_H, z_L, selected=S)
            probs = logits.softmax(-1)
            mixed = torch.where(given_mask, onehot_given.to(probs.dtype), probs)
            total_loss = total_loss + sudoku_constraint_loss(
                mixed[:, :, 1:self.cfg.vocab_size], self._constraint_mode)

    loss = total_loss / T
```

Key details:
- `z_H` and `z_L` are **detached** by the model's forward (`.detach()`) — no BPTT across ACT steps
- within each ACT step, gradients flow through the **last H_cycle** of IR only (first H_cycle-1 are `no_grad`)
- `selected` mask (30% random subset of free cells) acts as a focus signal via `injection`
- given cells are pinned to one-hot in `mixed` before computing constraint loss

## Validation

Single pass (no ACT loop), all free cells selected:

```python
# wrapper.py — validation_step
def validation_step(self, batch, batch_idx):
    x_ans_soft = _soft_init(q_in, self.cfg.vocab_size)
    logits, _, _ = self.model(q_in, x_ans_soft, selected=free_flat)  # z_H=None, z_L=None
    probs = logits.softmax(-1)
    mixed = torch.where(given_mask, onehot_given, probs)
    val_loss = sudoku_constraint_loss(mixed[:, :, 1:self.cfg.vocab_size], self._constraint_mode)
```

## Inference

### Single model

```python
# wrapper.py — infer()
def infer(self, q_in, n_steps=1):
    x_ans_soft = _soft_init(q_in, self.cfg.vocab_size)
    z_H, z_L = None, None
    for _ in range(n_steps):
        S = random 30% of free cells (all free on last step if n_steps=1)
        logits, z_H, z_L = self.model(q_in, x_ans_soft, z_H, z_L, selected=S)
    preds = logits[:, :, 1:V].argmax(-1) + 1  # hard assignment
    # fill free cells with predictions
```

### Ensemble (eval_ensemble.py)

Two models trained with different constraint modes. Combined at inference:

```
┌──────────────────────────────────────────────────────────────────┐
│ refine(model_rc, model_box, q_in, combine_fn, n_steps=10)       │
│                                                                  │
│   x_ans_soft = _soft_init(q_in)                                  │
│                                                                  │
│   for step in range(10):                                         │
│     S = 30% random free cells (all free on last step)            │
│     logits_rc  = model_rc(q_in, x_ans_soft, z_H=None, S)        │
│     logits_box = model_box(q_in, x_ans_soft, z_H=None, S)       │
│                                                                  │
│     combined = combine_fn(logits_rc, logits_box)                 │
│     probs = combined.softmax(-1)                                 │
│     x_ans_soft[S] = probs[S]      ◄── update soft board          │
│                                                                  │
│   preds = x_ans_soft[:,:,1:10].argmax(-1) + 1                   │
└──────────────────────────────────────────────────────────────────┘
```

Combination strategies:

```python
# eval_ensemble.py
def ensemble_confidence(logits_rc, logits_box):
    # Per-cell, pick the model with higher top1-top2 logit margin
    margin_rc  = logit_margin(logits_rc)
    margin_box = logit_margin(logits_box)
    pick_box = (margin_box > margin_rc).unsqueeze(-1)
    return torch.where(pick_box, logits_box, logits_rc)

def ensemble_average(logits_rc, logits_box):
    return (logits_rc + logits_box) / 2
```

Note: ensemble inference does **not** carry z_H/z_L across steps — each step starts fresh (`z_H=None`). The soft board `x_ans_soft` is the only state that persists across refinement steps.

## Optimizer: Muon + AdamW Split

```python
# wrapper.py — configure_optimizers
# 2D params (weight matrices) → Muon (momentum-based, orthogonal updates)
# Other params (biases, norms, embeddings) → AdamW
for p in self.model.parameters():
    (muon_params if p.ndim == 2 else adamw_params).append(p)

# LR schedule: linear warmup → cosine decay to 10% of peak
def lr_fn(step):
    if step < warmup: return step / warmup
    progress = (step - warmup) / (total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))
```

## Building Blocks Reference

### TransformerBlock

```python
# models/trm_blocks.py
def forward(self, x, film_cond=None, cross_attn_ctx=None):
    x = x + self.self_attn(rms_norm(x))         # pre-norm self-attention
    if self.cross_attn and cross_attn_ctx:
        x = x + self.cross_attn(rms_norm(x), cross_attn_ctx)
    x = x + self.mlp(rms_norm(x))               # SwiGLU FFN
    if self.film and film_cond:
        x = self.film(x, film_cond)              # FiLM: gamma*x + beta
    return x
```

### ResidualBlock (conv)

```python
# models/trm_blocks.py
def forward(self, x, film_cond=None):
    h = self.conv1(F.silu(self.norm1(x)))        # GroupNorm → SiLU → Conv3x3
    h = self.conv2(F.silu(self.norm2(h)))        # GroupNorm → SiLU → Conv3x3
    x = x + h                                    # residual
    if self.use_film and self.cond_mod and film_cond:
        x = self.cond_mod(x, film_cond)          # FiLM on residual stream
    return x
```

### LLevel (shared by both architectures)

```python
# Both TransformerLLevel and UNetLLevel follow the same pattern:
def forward(self, h, injection, **kwargs):
    h = h + injection                            # add focus signal
    for block in self.blocks:
        h = block(h, **kwargs)
    return h
```

## Tensor Shapes Cheat Sheet

```
Batch input:
  q_in:            (B, 9, 9)       integer grid, 0=empty, 1-9=digits
  q_out:           (B, 9, 9)       solution (ground truth)
  x_ans_soft:      (B, 81, 11)     soft probability board (V+1 classes)
  free_flat:       (B, 81)         bool mask of empty cells
  selected:        (B, 81)         bool mask of cells to focus on this step

DiT internal:
  z_H:             (B, 162, d)     [question_emb(81) | answer_emb(81)]
  z_L:             (B, 162, d)     scratchpad (zeros on first call)
  injection:       (B, 162, d)     selected_emb at answer positions

UNet internal:
  z_H:             (B, ch, 9, 9)   spatial feature map
  z_L:             (B, ch, 9, 9)   scratchpad (zeros on first call)
  injection:       (B, ch, 9, 9)   convolved selection mask

Output:
  logits:          (B, 81, 11)     per-cell class logits

Constraint loss input:
  mixed[:,:,1:10]: (B, 81, 9)      digit probabilities only (skip empty class)
```
