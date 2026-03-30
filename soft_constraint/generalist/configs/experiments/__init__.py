from .neutral_atoms import apply_neutral_atoms_config, inject_shared_properties as na_inject
from .single_map import apply_single_map_config, inject_shared_properties as sm_inject

DEFAULT_EXPERIMENT = 'neutral_atoms'

EXPERIMENTS = {
    'neutral_atoms': (apply_neutral_atoms_config, na_inject),
    'single_map': (apply_single_map_config, sm_inject),
}
