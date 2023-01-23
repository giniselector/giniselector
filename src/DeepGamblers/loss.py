import torch
import torch.nn.functional as F
from torch import Tensor


def deep_gamblers_loss(outputs: Tensor, targets: Tensor, reward: float = 1.0):
    outputs = F.softmax(outputs, dim=1)
    outputs, reservation = outputs[:, :-1], outputs[:, -1]
    gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
    doubling_rate = (gain + reservation / reward).log()

    loss = -doubling_rate.mean()

    return loss
