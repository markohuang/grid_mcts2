from .na_dit import NADiTModel

BACKBONES = {
    'trm_dit': NADiTModel,
}

def get_backbone(model_type, model_cfg):
    if model_type not in BACKBONES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(BACKBONES.keys())}")
    return BACKBONES[model_type](model_cfg)
