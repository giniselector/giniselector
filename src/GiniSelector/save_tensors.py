import logging
import os

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import torch.utils.data
from pytorch_lightning import loggers as pl_loggers

from src.GiniSelector.model_wrapper import ModelWrapper
from src.GiniSelector.train import Arguments, save_tensors
from src.utils.datamodules import DefaultDataModule
from src.utils.models import get_model_essentials

from .scores import gini

logger = logging.getLogger(__name__)


def main(args: Arguments):
    # Reproducibility
    pl.seed_everything(args.seed, workers=True)

    # pre-trained model
    model_essentials = get_model_essentials(args.model_name)
    model = model_essentials.pop("model")
    best_weights = torch.load(os.path.join(args.output_dir, "best.pth"), map_location="cpu")
    model.load_state_dict(best_weights)
    model = ModelWrapper(
        **model_essentials,
        model=model,
        root=args.checkpoints_dir,
        model_name=args.model_name,
        score_fn=gini,
        optimizer_kwargs=args.optimizer_kwargs,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
    )

    # datamodule
    datamodule = DefaultDataModule(
        args.dataset_name,
        args.data_dir,
        args.num_workers,
        args.val_split,
        args.batch_size,
        args.seed,
        args.pin_memory,
        train_shuffle=False,
        train_transforms=model_essentials["test_transforms"],
        test_transforms=model_essentials["test_transforms"],
    )

    # predictions
    callbacks = [pl_callbacks.TQDMProgressBar(refresh_rate=10)]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output_dir, name="logs")
    predictor = pl.Trainer(
        default_root_dir=args.output_dir,
        logger=tb_logger,
        accelerator="auto",
        enable_checkpointing=False,
        auto_select_gpus=True,
        callbacks=callbacks,  # type: ignore
        benchmark=True,
        max_epochs=1,
    )

    datamodule.setup()
    # train logits
    outputs = predictor.predict(model, dataloaders=[datamodule.train_dataloader()], return_predictions=True)
    logits, targets, scores = save_tensors(outputs, args.output_dir, "train")
    model_accuracy = (torch.argmax(logits, 1) == targets).float().mean().item()
    logger.info("Train logits shape: %s", logits.shape)
    logger.info("Train targets shape: %s", targets.shape)
    logger.info("Train Accuracy: %s", model_accuracy)

    # test logits
    outputs = predictor.predict(model, dataloaders=[datamodule.test_dataloader()], return_predictions=True)
    logits, targets, scores = save_tensors(outputs, args.output_dir, "test")
    model_accuracy = (torch.argmax(logits, 1) == targets).float().mean().item()
    logger.info("Test logits shape: %s", logits.shape)
    logger.info("Test targets shape: %s", targets.shape)
    logger.info("Train Accuracy: %s", model_accuracy)


if __name__ == "__main__":
    args = Arguments().parse_args()

    logging.basicConfig(
        format="---> %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info(args)
    main(args)
