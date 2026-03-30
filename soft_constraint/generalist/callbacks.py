"""Lightning callbacks for neutral atoms IR training."""
import json
from datetime import datetime
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger


class RegistryCallback(Callback):

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        cfg = pl_module.cfg
        self._log_to_registry(trainer, cfg)

    def _log_to_registry(self, trainer, cfg):
        registry_path = Path(cfg.save_dir) / 'run_registry.jsonl'

        metrics = {}
        for key in ['val_loss', 'val_true_cost', 'train_loss']:
            val = trainer.callback_metrics.get(key)
            if val is not None:
                metrics[key] = round(float(val), 4)

        aug_cfg = cfg.data.augmentation
        augmentations = list(aug_cfg.transforms) if aug_cfg.enabled else []

        entry = {
            'run_id': cfg.run_id,
            'experiment': cfg.experiment,
            'steps': trainer.global_step,
            'model_type': cfg.model_type,
            'model_variant': cfg.model_variant,
            'augmentations': augmentations,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        with open(registry_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        logger.info(f"Logged run {cfg.run_id} to registry")


def get_default_callbacks(cfg):
    return [
        RegistryCallback(),
    ]
