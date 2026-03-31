from ml_collections import ConfigDict
from .base import get_base_config
from .model_configs import MODEL_CONFIGS
from .experiments import EXPERIMENTS, DEFAULT_EXPERIMENT


def get_config(config_string=None):
    """
    Parameterized config dispatcher.

    config_string formats:
        "model_type/variant"              -> default experiment (neutral_atoms)
        "experiment/model_type/variant"   -> explicit experiment
        "model_type"                      -> default experiment, variant='iter_T1N4'

    CLI usage:
        python main2.py --cfg=config2.py:trm_dit/iter_T1N4
        python main2.py --cfg=config2.py:trm_dit/tiny_T2N2 --cfg.optim.lr=1e-4
    """
    experiment, model_type, variant = _parse_config_string(config_string)

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from {list(EXPERIMENTS.keys())}")

    # Layer 1: universal defaults
    cfg = get_base_config()

    # Layer 2: experiment-specific config
    apply_experiment, inject_fn = EXPERIMENTS[experiment]
    apply_experiment(cfg)

    # Layer 3: model config
    cfg.model_type = model_type
    cfg.model_variant = variant
    cfg.model = MODEL_CONFIGS[model_type](variant)

    # Layer 4: cross-injection (experiment tells model what it needs to know)
    inject_fn(cfg)

    return cfg


def _parse_config_string(config_string):
    if not config_string:
        return DEFAULT_EXPERIMENT, 'trm_dit', 'iter_T1N4'

    parts = config_string.split('/')

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        if parts[0] in EXPERIMENTS and parts[1] in MODEL_CONFIGS:
            return parts[0], parts[1], 'iter_T1N4'
        elif parts[0] in MODEL_CONFIGS:
            return DEFAULT_EXPERIMENT, parts[0], parts[1]
        else:
            raise ValueError(
                f"Cannot parse '{config_string}'. "
                f"Experiments: {list(EXPERIMENTS.keys())}, Models: {list(MODEL_CONFIGS.keys())}"
            )
    elif len(parts) == 1:
        if parts[0] in MODEL_CONFIGS:
            return DEFAULT_EXPERIMENT, parts[0], 'iter_T1N4'
        raise ValueError(f"Unknown model_type: {parts[0]}")
    else:
        raise ValueError(f"Config string has too many parts: '{config_string}'")
