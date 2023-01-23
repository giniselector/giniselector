import torch
from torch import Tensor


def soft_coverage(scores: Tensor) -> Tensor:
    return (1 - scores).mean()


def hard_coverage(scores: Tensor, thr: float) -> Tensor:
    return (scores <= thr).float().mean()


def soft_risk(
    scores: Tensor,
    logits: Tensor,
    targets: Tensor,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
) -> Tensor:
    return (soft_coverage(scores) * loss_fn(logits, targets)).mean() / soft_coverage(scores)


def soft_risk_multi_class_classif(scores: Tensor, logits: Tensor, targets: Tensor):
    return soft_risk(scores, logits, targets, torch.nn.CrossEntropyLoss(reduction="none"))


def hard_risk(
    scores: Tensor,
    logits: Tensor,
    targets: Tensor,
    thr: float,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
):
    return (hard_coverage(scores, thr) * loss_fn(logits, targets)).mean() / hard_coverage(scores, thr)


def selective_net_risk(scores: Tensor, pred: Tensor, targets: Tensor, thr: float):
    covered_idx = scores <= thr
    return torch.sum(pred[covered_idx] != targets[covered_idx]) / torch.sum(covered_idx)


def risks_coverages(
    scores: Tensor,
    logits: Tensor,
    targets: Tensor,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
):
    risks = []
    coverages = []
    thrs = []
    for thr in scores.unique():
        risks.append(hard_risk(scores, logits, targets, thr, loss_fn))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = torch.tensor(risks).float()
    coverages = torch.tensor(coverages).float()
    thrs = torch.tensor(thrs).float()
    return risks, coverages, thrs


def risks_coverages_selective_net(scores: Tensor, pred: Tensor, targets: Tensor, sort: bool = True):
    """
    Returns:

        risks, coverages, thrs
    """
    # this function is slow
    risks = []
    coverages = []
    thrs = []
    for thr in scores.unique():
        risks.append(selective_net_risk(scores, pred, targets, thr))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = torch.tensor(risks).float()
    coverages = torch.tensor(coverages).float()
    thrs = torch.tensor(thrs).float()

    # sort by coverages
    if sort:
        sorted_idx = torch.argsort(coverages)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]
    return risks, coverages, thrs
