import json
import logging
import os
import sys

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch
import torch.utils.data
import torchmetrics.functional as metrics
from pytorch_lightning import loggers as pl_loggers
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.ConfidNet.loss import confid_mse_loss
from src.ConfidNet.model_wrapper import ConfidNet, ModelWrapper
from src.utils import helpers
from src.utils.argparser import Arguments as BaseArguments
from src.utils.datamodules import get_vision_datamodule_cls
from src.utils.models import get_model_essentials
from src.utils.optimizers import get_optimizer_cls, get_scheduler_cls

logger = logging.getLogger(__name__)


def save_tensors(outputs, root, prefix="train"):
    logits = torch.cat([x["logits"] for x in outputs], dim=0)
    targets = torch.cat([x["targets"] for x in outputs], dim=0).squeeze()
    scores = torch.cat([x["scores"] for x in outputs], dim=0).squeeze()

    torch.save(logits, os.path.join(root, f"{prefix}_logits.pt"))
    torch.save(targets, os.path.join(root, f"{prefix}_targets.pt"))
    torch.save(scores, os.path.join(root, f"{prefix}_scores.pt"))

    return logits, targets, scores


class Arguments(BaseArguments):
    config_file: str = "src/ConfidNet/confidnet.json"

    @property
    def output_dir(self):
        if self._output_dir is None:
            path = os.path.join(self.checkpoints_dir, "confid_net", self.model_name, str(self.seed))
            if self.val_split > 0:
                path = os.path.join(path, f"val_split_{self.val_split}")
            return path
        return self._output_dir


def main(args: Arguments):
    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pre trained model
    criterion = confid_mse_loss
    optimizer_cls = get_optimizer_cls(args.optimizer)
    lr_scheduler_cls = get_scheduler_cls(args.lr_scheduler)
    model_essentials = get_model_essentials(args.model_name)
    model_arch = model_essentials["model"]
    model = ModelWrapper(
        model=model_arch,
        input_dim=model_essentials["input_dim"],
        criterion=criterion,
        optimizer_cls=optimizer_cls,
        lr_scheduler_cls=lr_scheduler_cls,
        optimizer_kwargs=args.optimizer_kwargs,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
    )
    # load checkpoint
    weights = os.path.join(args.checkpoints_dir, "ce", args.model_name, str(args.seed), "best.pth")
    checkpoint = torch.load(weights, map_location="cpu")
    model.model.load_state_dict(checkpoint)
    model.freeze()
    model.freeze_bn()
    model.to(device)

    # check that the model is frozen
    for param in model.parameters():
        assert not param.requires_grad

    # data
    logger.info(args.dataset_name)
    datamodule_cls = get_vision_datamodule_cls(args.datamodule)
    datamodule = datamodule_cls(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed,
        pin_memory=args.pin_memory,
        train_shuffle=True,
        train_transforms=model_essentials["train_transforms"],
        val_transforms=model_essentials["test_transforms"],
        test_transforms=model_essentials["test_transforms"],
    )
    datamodule.setup()

    # replace model with confidnet model
    input_example = torch.rand(1, *model_essentials["input_dim"]).to(device)
    features_nodes = model_essentials["features_nodes"]
    feature_extractor = create_feature_extractor(model_arch, features_nodes)
    features_dim = feature_extractor(input_example)["features"].shape[-1]
    confid_model = ConfidNet(feature_extractor, features_dim)
    model.model = confid_model
    model.to(device)

    # check accuracy
    for _ in range(2):
        test_loader = datamodule.test_dataloader()
        test_logits = []
        test_labels = []
        model.eval()
        with torch.no_grad():
            for i, (x, t) in enumerate(tqdm(test_loader)):
                x = x.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                out_class, _ = model(x)

                test_logits.append(out_class.detach().cpu())
                test_labels.append(t.detach().cpu())

        test_logits = torch.cat(test_logits, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        test_pred = torch.argmax(test_logits, dim=1)
        print(test_logits.shape, test_pred.shape, test_labels.shape)
        test_acc = metrics.accuracy(test_pred, test_labels).item()
        logger.info("Test accuracy before training %s", test_acc)

    # training
    callbacks = [
        pl_callbacks.TQDMProgressBar(refresh_rate=100),
        pl_callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output_dir, name="logs", log_graph=True)
    callbacks.append(pl_callbacks.ModelCheckpoint(dirpath=args.output_dir, filename="best", save_last=True))
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        logger=tb_logger,
        accelerator="auto",
        enable_checkpointing=args.enable_checkpointing,
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        callbacks=callbacks,
        auto_scale_batch_size=True if args.batch_size is None else False,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=1,
        max_epochs=args.epochs if not args.debug else 1,
        fast_dev_run=True if args.debug else False,
        overfit_batches=1 if args.debug else 0,
        profiler="simple" if args.debug else None,
    )
    trainer.fit(model, datamodule=datamodule)
    if args.debug:
        sys.exit(0)

    # test best model and save only model weights
    if trainer.checkpoint_callback is not None:
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model_score = trainer.checkpoint_callback.best_model_score
        print("Best model path:", best_model_path)
        print("Best model score:", best_model_score)
        if best_model_path != "":
            chkpt = torch.load(best_model_path)
            # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict',
            # 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name',
            # 'hyper_parameters'])
            model.load_state_dict(chkpt["state_dict"])
            # save best weights
            state_dict = {k.replace("model.", ""): v for k, v in chkpt["state_dict"].items()}
            torch.save(state_dict, os.path.join(args.output_dir, "best.pth"))

        # save last model weights
        last_model_path = trainer.checkpoint_callback.last_model_path
        print("Last model path:", last_model_path)
        if last_model_path != "":
            chkpt = torch.load(last_model_path)
            model.load_state_dict(chkpt["state_dict"])
            # save best weights
            state_dict = {k.replace("model.", ""): v for k, v in chkpt["state_dict"].items()}
            torch.save(state_dict, os.path.join(args.output_dir, "last.pth"))

    best_weights = torch.load(os.path.join(args.output_dir, "best.pth"))
    confid_model.load_state_dict(best_weights)
    model.model = confid_model
    test_results = trainer.test(model, datamodule=datamodule)

    with open(os.path.join(args.output_dir, "logs", "test_results.json"), "w", encoding="utf8") as f:
        json.dump(test_results, f, indent=2)

    # save results
    results_obj = {
        "model": args.model_name,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "optimizer_kwargs": args.optimizer_kwargs,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_kwargs": args.lr_scheduler_kwargs,
        "epochs": args.epochs,
        "test_results": test_results,
    }
    helpers.append_results_to_file(results_obj, filename=args.results_filename)

    # save logits, targets, and scores
    outputs = trainer.predict(model, dataloaders=[datamodule.test_dataloader()], return_predictions=True)
    save_tensors(outputs, args.output_dir, "test")


if __name__ == "__main__":
    args = Arguments().parse_args()

    logging.basicConfig(
        format="---> %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info(args)
    main(args)
