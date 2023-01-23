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

from src.ConfidNet.loss import confid_mse_loss
from src.utils import eval as detect_eval

logger = logging.getLogger(__name__)


class ConfidNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module, features_dim: int) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

        # freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # disable bn
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False

        # disable dropout
        self.feature_extractor.eval()

        # confid net module
        self.confidnet = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )

    def forward(self, x: Tensor):
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor(x)
        uncertainty = self.confidnet(x["features"])
        return x["linear"], uncertainty


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        input_dim: Tuple,
        url: Optional[str] = None,
        criterion: Callable = confid_mse_loss,
        optimizer_cls: Type[optim.Optimizer] = optim.SGD,
        lr_scheduler_cls: Type[torch_lr_scheduler._LRScheduler] = torch_lr_scheduler.CosineAnnealingLR,
        optimizer_kwargs: Dict[str, Any] = {},
        lr_scheduler_kwargs: Dict[str, Any] = {},
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.input_dim = input_dim
        self.url = url

        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls

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

    def _eval_batch(self, outs, y, prefix="train"):
        loss = self.criterion(*outs, y)
        acc = torchmetrics.functional.accuracy(outs[0], y)
        score = -outs[1].squeeze()

        results = {f"{prefix}/loss": loss, f"{prefix}/acc": acc}
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
        return outs
        # return {"prediction_out": outs[0], "confidence_out": outs[1]}

    def training_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        logits = outs[0]
        pred = torch.argmax(logits, dim=1)
        loss, score = self._eval_batch(outs, y, "train")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]):
        self._eval_epoch(outputs, "train")

    def validation_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        pred = torch.argmax(outs[0], dim=1)
        loss, score = self._eval_batch(outs, y, "val")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self._eval_epoch(outputs, "val")

    def test_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        outs = self.forward(x)
        pred = torch.argmax(outs[0], dim=1)
        loss, score = self._eval_batch(outs, y, "test")
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

    def freeze_bn(self):
        # disable bn
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False
