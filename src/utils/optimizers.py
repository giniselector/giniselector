from typing import Type

from torch import nn, optim

criterion_registry = {
    "crossentropy": nn.CrossEntropyLoss(),
    "bce": nn.BCEWithLogitsLoss(),
}

optimizers_registry = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "none": None,
}

schedulers_registry = {
    "step": optim.lr_scheduler.StepLR,
    "multi_step": optim.lr_scheduler.MultiStepLR,
    "cosine_annealing": optim.lr_scheduler.CosineAnnealingLR,
    "none": None,
}


def get_criterion(criterion_name: str) -> nn.modules.loss._Loss:
    return criterion_registry.get(criterion_name.lower(), None)


def get_optimizer_cls(optimizer_name: str) -> Type[optim.Optimizer]:
    return optimizers_registry.get(optimizer_name.lower(), None)


def get_scheduler_cls(scheduler_name: str) -> Type[optim.lr_scheduler._LRScheduler]:
    return schedulers_registry.get(scheduler_name.lower(), None)
