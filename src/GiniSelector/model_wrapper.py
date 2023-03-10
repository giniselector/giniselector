import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler as torch_lr_scheduler
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from src.utils import eval as detect_eval

logger = logging.getLogger(__name__)


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        root: str,
        model_name: str,
        input_dim: Tuple,
        score_fn: Optional[Callable] = None,
        features_nodes: Optional[List[str]] = None,
        test_transforms: Optional[Type[transforms.Compose]] = None,
        train_transforms: Optional[Type[transforms.Compose]] = None,
        download: bool = True,
        url: Optional[str] = None,
        criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
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
        self.features_nodes = features_nodes
        self.test_transforms = test_transforms
        self.train_transforms = train_transforms
        self.download = download
        self.url = url
        self.root = root
        self.model_name = model_name
        self.score_fn = score_fn

        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls

        self.feature_extractor = None

        self.example_input_array = (
            torch.randn(1, *self.input_dim).to(self.device, dtype=torch.float32) if self.input_dim is not None else None
        )

        self.save_hyperparameters(ignore=["model", "criterion"])

        self.__post_init__()

    def __post_init__(self):
        self.setup()

        if self.features_nodes is not None:
            self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)

    def setup(self, stage=None):
        # download pre trained weights etc
        logger.info(f"{self.root} {self.model_name}")
        model_dir = os.path.join(self.root, self.model_name)

        os.makedirs(model_dir, exist_ok=True)
        if self.url is not None:
            parts = torch.hub.urlparse(self.url)
            filename = os.path.basename(parts.path)

            # download from url
            cached_file = os.path.join(model_dir, filename)
            if not os.path.exists(cached_file) and self.download:
                logger.info('Downloading: "{}" to {}\n'.format(self.url, cached_file))
                torch.hub.download_url_to_file(self.url, cached_file)

            # load from memory
            cached_file = os.path.join(model_dir, parts.path)
            if os.path.exists(cached_file):
                logger.info("loading weights from chached file: %s", cached_file)
                w = torch.load(cached_file, map_location=self.device)
                self.model.load_state_dict(w)
            else:
                raise FileNotFoundError("Cached file not found: {}".format(cached_file))

    def on_fit_start(self):
        if self.input_dim is not None:
            tensorboard_logger = self.logger.experiment

            prototype_array = torch.randn(1, *self.input_dim).to(self.device)
            tensorboard_logger.add_graph(self.model, prototype_array)

    def _batch_extraction(self, batch):
        x, y = batch
        return x, y

    def _eval_batch(self, logits, y, prefix="train"):
        loss = self.criterion(logits, y)
        acc = torchmetrics.functional.accuracy(logits, y)
        score = torch.ones_like(y)
        if self.score_fn is not None:
            score = self.score_fn(logits)

        results = {f"{prefix}/loss": loss, f"{prefix}/acc": acc}
        self.log_dict(results, prog_bar=True, on_epoch=True, on_step=False, rank_zero_only=True)
        return loss, acc, score

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

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        logits = self.forward(x)
        pred = torch.argmax(logits, dim=1)
        loss, acc, score = self._eval_batch(logits, y, "train")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]):
        self._eval_epoch(outputs, "train")

    def validation_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        logits = self.forward(x)
        pred = torch.argmax(logits, dim=1)
        loss, acc, score = self._eval_batch(logits, y, "val")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self._eval_epoch(outputs, "val")

    def test_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        logits = self.forward(x)
        pred = torch.argmax(logits, dim=1)
        loss, acc, score = self._eval_batch(logits, y, "test")
        return {"loss": loss, "pred": pred, "y": y, "score": score}

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self._eval_epoch(outputs, "test")

    def predict_step(self, batch, batch_idx):
        x, y = self._batch_extraction(batch)
        logits = self.forward(x)
        scores = torch.ones_like(y)
        if self.score_fn is not None:
            scores = self.score_fn(logits)

        return {"logits": logits, "targets": y, "scores": scores}

    def extract_features(self, x: Tensor) -> Dict[str, Tensor]:
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            with torch.no_grad():
                return self.feature_extractor(x)
        else:
            raise ValueError("Feature extractor is not defined. Please define the `features_nodes` argument.")

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
