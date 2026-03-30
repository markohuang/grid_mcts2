from ml_collections import ConfigDict

def get_base_config():
    cfg = ConfigDict()
    cfg.run_id = None
    cfg.run_dir = None
    cfg.seed = 42
    cfg.save_dir = './outputs'
    cfg.accumulate_grad_batches = 1

    # Optimizer
    cfg.optim = ConfigDict()
    cfg.optim.type = 'muon'
    cfg.optim.lr = 3e-4
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.999
    cfg.optim.eps = 1e-8
    cfg.optim.weight_decay = 0.01
    cfg.optim.warmup_steps = 1000
    cfg.optim.muon_momentum = 0.95

    # Trainer
    use_muon = cfg.optim.type == 'muon'
    cfg.trainer = ConfigDict()
    cfg.trainer.max_steps = 50000
    cfg.trainer.check_val_every_n_epoch = None
    cfg.trainer.val_check_interval = 1000
    cfg.trainer.limit_val_batches = 50
    cfg.trainer.log_every_n_steps = 10
    cfg.trainer.precision = 'bf16-mixed'
    cfg.trainer.gradient_clip_val = 0 if use_muon else 1.0
    cfg.trainer.accelerator = 'auto'
    cfg.trainer.devices = 1

    # Early stopping
    cfg.early_stopping = ConfigDict()
    cfg.early_stopping.monitor = 'val_loss'
    cfg.early_stopping.patience = 10
    cfg.early_stopping.mode = 'min'
    cfg.early_stopping.verbose = False

    # W&B
    cfg.wandb = ConfigDict()
    cfg.wandb.project = 'neutral-atoms-ir'

    return cfg
