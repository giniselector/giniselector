import torch
from torch import Tensor


def hard_coverage(scores: Tensor, thr: float) -> Tensor:
    return (scores <= thr).float().mean()


def selective_net_risk(scores: Tensor, pred: Tensor, targets: Tensor, thr: float):
    covered_idx = scores <= thr
    return torch.sum(pred[covered_idx] != targets[covered_idx]) / torch.sum(covered_idx)


def risks_coverages_selective_net(scores: Tensor, pred: Tensor, targets: Tensor, sort: bool = True):
    """
    Returns:

        risks, coverages, thrs
    """
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
