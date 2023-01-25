from typing import Callable

import torch
from torch import Tensor


def gini(logits: Tensor):
    g = torch.sum(torch.softmax(logits, 1) ** 2, 1)
    return 1 - g


scores_registry = {
    "gini": gini,
}


def get_score_fn(name: str) -> Callable:
    if name not in scores_registry:
        raise ValueError(f"Unknown score function {name}")
    return scores_registry[name]
