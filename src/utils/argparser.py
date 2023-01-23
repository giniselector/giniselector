import json
import os
from typing import Any, Dict, Optional

from tap import Tap

from src.utils import helpers

from . import config


class Arguments(Tap):
    # experiment
    _output_dir: Optional[str] = None
    debug: bool = False
    early_stopping: bool = False
    patience: int = 16
    enable_checkpointing: bool = True
    epochs: int = 300
    check_val_every_n_epoch: int = 5

    config_file: Optional[str] = None

    # model
    model_name: str = "densenet121_cifar10"
    optimizer: str = "sgd"
    criterion: str = "crossentropy"
    lr_scheduler: str = "cosine_annealing"
    optimizer_kwargs: Dict[str, Any] = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "nesterov": True,
    }
    lr_scheduler_kwargs: Dict[str, Any] = {"T_max": 280}

    # datamodule
    datamodule: str = "default"
    dataset_name: str = None  # type:ignore
    data_dir: str = config.DATA
    checkpoints_dir: str = config.CHECKPOINTS
    num_workers: int = 4
    val_split: float = 0
    batch_size: int = 64
    seed: int = 42
    pin_memory: bool = True

    def configure(self) -> None:
        # this is ran first
        self.add_argument(
            "--optimizer_kwargs", type=helpers.str_to_dict, help="""e.g --optimizer_kwargs '{"lr": 0.1}'"""
        )
        self.add_argument("--lr_scheduler_kwargs", type=helpers.str_to_dict)

    def process_args(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.dataset_name is None:
            self.dataset_name = self.model_name.split("_")[1].lower()

        # load from config file
        if self.config_file is not None:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)

    @property
    def output_dir(self):
        if self._output_dir is None:
            path = os.path.join(self.checkpoints_dir, self.model_name, str(self.seed))
            if self.val_split > 0:
                path = os.path.join(path, f"val_split_{self.val_split}")
            return path
        return self._output_dir

    @property
    def results_filename(self) -> str:
        return os.path.join(self.checkpoints_dir, "results.csv")
