from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def selective_net_loss(
    prediction_out: Tensor,
    selection_out: Tensor,
    aux_out: Tensor,
    target: Tensor,
    target_coverage: float,
    lm: float = 32.0,
    alpha: float = 0.5,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Args:
        prediction_out: (B, num_classes)
        selection_out:  (B, 1)
        aux_out:        (B, num_classes)
        target:         (B)
        coverage: target coverage.
        lm: Lagrange multiplier for coverage constraint. Original experiment value is 32.
    """
    # compute empirical coverage (=phi^)
    empirical_coverage = selection_out.mean()

    # compute empirical risk (=r^)
    emprical_risk = (F.cross_entropy(prediction_out, target, reduction="none") * selection_out.view(-1)).mean()
    emprical_risk = emprical_risk / empirical_coverage

    # compute penalty (=psi)
    penalty = (
        torch.max(
            target_coverage - empirical_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=empirical_coverage.device),
        )
        ** 2
    )
    penalty *= lm

    selective_loss = emprical_risk + penalty

    # standard cross entropy loss for the auxiliary network
    ce = F.cross_entropy(aux_out, target)

    # total loss
    total_loss = alpha * selective_loss + (1 - alpha) * ce

    # loss information dict
    loss_dict = {}
    loss_dict["empirical_coverage"] = empirical_coverage.detach().cpu().item()
    loss_dict["emprical_risk"] = emprical_risk.detach().cpu().item()
    loss_dict["penalty"] = penalty.detach().cpu().item()
    loss_dict["selective_loss"] = selective_loss.detach().cpu().item()
    loss_dict["ce"] = ce.detach().cpu().item()
    loss_dict["loss"] = total_loss.detach().cpu().item()

    return total_loss, loss_dict
