import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler as torch_lr_scheduler

from src.SelectiveNet.loss import selective_net_loss
from src.utils import eval as detect_eval

logger = logging.getLogger(__name__)


class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.
    """

    def __init__(self, features, dim_features: int, num_classes: int, init_weights=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.
            num_classes: number of classification class.
        """
        super().__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes
        # represented as f() in the original paper
        self.classifier = torch.nn.Linear(self.dim_features, self.num_classes)

        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid(),
        )

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Linear(self.dim_features, self.num_classes)

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, x):
        x = self.features(x)
        if isinstance(x, dict):
            x = x["features"]
        x = x.view(x.size(0), -1)

        prediction_out = self.classifier(x)
        selection_out = self.selector(x)
        auxiliary_out = self.aux_classifier(x)

        return (prediction_out, selection_out, auxiliary_out)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        target_coverage: float,
        input_dim: Tuple,
        url: Optional[str] = None,
        criterion: Callable = selective_net_loss,
        optimizer_cls: Type[optim.Optimizer] = optim.SGD,
        lr_scheduler_cls: Type[torch_lr_scheduler._LRScheduler] = torch_lr_scheduler.CosineAnnealingLR,
        optimizer_kwargs: Dict[str, Any] = {},
        lr_scheduler_kwargs: Dict[str, Any] = {},
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.target_coverage = target_coverage
        self.input_dim = input_dim
        self.url = url

        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls

        self.feature_extractor = None

        self.example_input_array = (
            torch.randn(1, *self.input_dim).to(self.device, dtype=torch.float32) if self.input_dim is not None else None
        )

        self.save_hyperparameters(ignore=["model", "criterion"])

    def on_fit_start(self):
        if self.input_dim is not None:
            tensorboard_logger = self.logger.experiment

            prototype_array = torch.randn(1, *self.input_dim).to(self.device)
            tensorboard_logger.add_graph(self.model, prototype_array)

    def _batch_extraction(self, batch):
        x, y = batch
        return x, y

    def _eval_batch(self, outs, y, target_coverage, prefix="train"):
        loss, loss_dict = self.criterion(*outs, y, target_coverage)
        acc = torchmetrics.functional.accuracy(outs[0], y)
        score = -outs[1].squeeze()

        results = {f"{prefix}/loss": loss, f"{prefix}/acc": acc}
        results.update({f"{prefix}-sn/{k}": v for k, v in loss_dict.items()})
        self.log_dict(results, prog_bar=True, on_epoch=True, on_step=False, rank_zero_only=True)
        return loss, score

    def _eval_epoch(self, outputs, prefix="train"):
        preds = torch.cat([x["pred"] for x in outputs])
        targets = torch.cat([x["y"] for x in outputs])
        scores = torch.cat([x["score"] for x in outputs])
        # random calibration and  test split with 10% of the data
        cal_indexes, test_indexes = train_test_split(
            np.arange(len(scores)), test_size=0.9, random_state=42, shuffle=True
        )
        cal_risks, cal_coverages, cal_thrs = detect_eval.risks_coverages_selective_net(
            scores[cal_indexes], preds[cal_indexes], targets[cal_indexes]
        )

        target_coverages = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        coverages = {}
        risks = {}
        accs = {}
        for tc in target_coverages:
            _thr = cal_thrs[torch.searchsorted(cal_coverages, tc)].item()
            coverages[f"{prefix}/cov/{tc:.2f}"] = detect_eval.hard_coverage(scores[test_indexes], _thr).item()
            risks[f"{prefix}/risk/{tc:.2f}"] = (
                100
                * detect_eval.selective_net_risk(
                    scores[test_indexes], preds[test_indexes], targets[test_indexes], _thr
                ).item()
            )
            # accuracy
            coverage_filt = scores[test_indexes] <= _thr
            acc = torchmetrics.functional.accuracy(
                preds[test_indexes][coverage_filt],
                targets[test_indexes][coverage_filt],
            ).item()
            accs[f"{prefix}/acc/{tc:.2f}"] = acc

        results = {**coverages, **risks, **accs}
        self.log_dict(results, prog_bar=False, on_epoch=True, on_step=False, rank_zero_only=True)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any):
        outs = self.model(x, *args, **kwargs)
        assert outs[0].shape[-1] == self.model.num_classes
        return outs
        # return {"prediction_out": outs[0], "selection_out": outs[1], "auxiliary_out": outs[2]}

    def training_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        logits = outs[0]
        pred = torch.argmax(logits, dim=1)
        loss, score = self._eval_batch(outs, y, self.target_coverage, "train")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]):
        self._eval_epoch(outputs, "train")

    def validation_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        pred = torch.argmax(outs[0], dim=1)
        loss, score = self._eval_batch(outs, y, self.target_coverage, "val")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self._eval_epoch(outputs, "val")

    def test_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        pred = torch.argmax(outs[0], dim=1)
        loss, score = self._eval_batch(outs, y, self.target_coverage, "test")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self._eval_epoch(outputs, "test")

    def predict_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        logits = outs[0]
        scores = -outs[1]

        return {"logits": logits, "targets": y, "scores": scores}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.hparams.optimizer_kwargs,
        )  # type: ignore
        if self.lr_scheduler_cls is not None:
            lr_scheduler = self.lr_scheduler_cls(
                optimizer,
                **self.hparams.lr_scheduler_kwargs,
            )  # type: ignore
            return [optimizer], [lr_scheduler]
        return [optimizer]
