import torch
import random


def make_cell_permutation(board_h: int, board_w: int, transform: int) -> torch.Tensor:
    """Returns a permutation of flat cell indices for the given spatial transform.
    transform: 0=identity, 1=h-flip, 2=v-flip, 3=hv-flip,
               4=rot90, 5=rot90+h-flip, 6=rot90+v-flip, 7=rot90+hv-flip
    Transforms 4-7 only valid for square boards (board_h == board_w).
    """
    perm = torch.zeros(board_h * board_w, dtype=torch.long)
    for r in range(board_h):
        for c in range(board_w):
            nr, nc = r, c
            if transform >= 4:
                nr, nc = nc, board_h - 1 - nr  # rot90
            if (transform % 4) in (1, 3):
                nc = board_w - 1 - nc  # h-flip
            if (transform % 4) in (2, 3):
                nr = board_h - 1 - nr  # v-flip
            # For rot90, board dims swap: new_h=board_w, new_w=board_h
            if transform >= 4:
                perm[r * board_w + c] = nr * board_h + nc
            else:
                perm[r * board_w + c] = nr * board_w + nc
    return perm


def num_transforms(board_h: int, board_w: int) -> int:
    return 8 if board_h == board_w else 4


def augment_features_and_policy(features: torch.Tensor, policy: torch.Tensor,
                                 board_h: int, board_w: int, transform: int = None):
    """Augment a batch of (features, policy) with a spatial transform.
    features: (batch, num_tasks+1, board_size, num_qubits)
    policy: (batch, board_size)
    Returns augmented (features, policy) with same shapes.
    """
    if transform is None:
        transform = random.randint(0, num_transforms(board_h, board_w) - 1)
    if transform == 0:
        return features, policy
    perm = make_cell_permutation(board_h, board_w, transform).to(features.device)
    # Permute board_size dimension of features
    aug_features = features[:, :, perm, :]
    # Permute policy targets
    aug_policy = policy[:, perm]
    return aug_features, aug_policy
