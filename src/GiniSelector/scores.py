from typing import Callable

import torch
from torch import Tensor


def gini(logits: Tensor, temperature: float = 1):
    g = torch.sum(torch.softmax(logits / temperature, 1) ** 2, 1)
    return 1 - g


def msp(logits: Tensor, temperature: float = 1) -> Tensor:
    g = torch.softmax(logits / temperature, 1).max(1)[0]
    return 1 - g


def ce(logits: Tensor, *args, **kwargs) -> Tensor:
    predicted_classes = torch.argmax(logits, dim=1)
    return torch.nn.functional.cross_entropy(logits, predicted_classes, reduction="none")


scores_registry = {
    "msp": msp,
    "gini": gini,
    "ce": ce,
}


def get_score_fn(name: str) -> Callable:
    if name not in scores_registry:
        raise ValueError(f"Unknown score function {name}")
    return scores_registry[name]
