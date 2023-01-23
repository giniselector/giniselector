import torch
import torch.nn.functional as F
from torch import Tensor


def confid_mse_loss(logits: Tensor, confidence: Tensor, target: Tensor, weighting: float = 1):
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    confidence = torch.sigmoid(confidence).squeeze()
    # Apply optional weighting
    weights = torch.ones_like(target, dtype=torch.float32)
    weights[(probs.argmax(dim=1) != target)] *= weighting
    labels_hot = torch.eye(num_classes).to(target.device)[target]
    loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
    return torch.mean(loss)
