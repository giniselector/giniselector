import json
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics
from sklearn.model_selection import train_test_split
from torchvision.models.feature_extraction import create_feature_extractor

from src.ConfidNet.model_wrapper import ConfidNet, ModelWrapper
from src.ConfidNet.train import Arguments, save_tensors
from src.utils import eval as detect_eval
from src.utils import helpers
from src.utils.datamodules import get_vision_datamodule_cls
from src.utils.models import get_model_essentials

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(args: Arguments):
    model_essentials = get_model_essentials(args.model_name)
    model_arch = model_essentials["model"]
    features_nodes = model_essentials["features_nodes"]
    feature_extractor = create_feature_extractor(model_arch, features_nodes)
    input_example = torch.rand(1, *model_essentials["input_dim"])
    features_dim = feature_extractor(input_example)["features"].shape[-1]
    confid_model = ConfidNet(feature_extractor, features_dim)
    best_weights = torch.load(os.path.join(args.output_dir, "best.pth"))
    confid_model.load_state_dict(best_weights)

    model = ModelWrapper(model=confid_model, input_dim=model_essentials["input_dim"])

    return model


def main(args: Arguments):
    model_essentials = get_model_essentials(args.model_name)
    datamodule_cls = get_vision_datamodule_cls(name=args.datamodule)
    datamodule = datamodule_cls(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=1,
        pin_memory=args.pin_memory,
        train_shuffle=False,
        train_transforms=model_essentials["test_transforms"],
        val_transforms=model_essentials["test_transforms"],
        test_transforms=model_essentials["test_transforms"],
    )
    datamodule.setup()

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator="auto",
        auto_select_gpus=True,
        auto_scale_batch_size=True if args.batch_size is None else False,
        max_epochs=1,
        logger=False,
    )

    all_results = {}
    for seed in range(1, 11, 1):
        args.seed = seed
        pl.seed_everything(args.seed, workers=True)
        model = load_model_from_checkpoint(args)

        # outputs = trainer.predict(model, dataloaders=[datamodule.train_dataloader()], return_predictions=True)
        # save_tensors(outputs, args.output_dir, "train")
        outputs = trainer.predict(model, dataloaders=[datamodule.test_dataloader()], return_predictions=True)
        logits, targets, scores = save_tensors(outputs, args.output_dir, "test")
        preds = logits.argmax(1)
        print(f"acc: {torchmetrics.functional.accuracy(preds, targets).item():.2f}")

        # evaluation
        # random calibration and  test split with 10% of the data
        cal_indexes, test_indexes = train_test_split(
            np.arange(len(scores)), test_size=0.9, random_state=args.seed, shuffle=True
        )
        cal_risks, cal_coverages, cal_thrs = detect_eval.risks_coverages_selective_net(
            scores[cal_indexes], preds[cal_indexes], targets[cal_indexes]
        )

        coverages = {}
        risks = {}
        accs = {}
        target_coverages = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        for tc in target_coverages:
            _thr = cal_thrs[torch.searchsorted(cal_coverages, tc)].item()
            coverages[f"cov/{tc:.2f}"] = detect_eval.hard_coverage(scores[test_indexes], _thr).item()
            risks[f"risk/{tc:.2f}"] = (
                100
                * detect_eval.selective_net_risk(
                    scores[test_indexes], preds[test_indexes], targets[test_indexes], _thr
                ).item()
            )
            # accuracy
            coverage_filt = scores[test_indexes] <= _thr
            acc = torchmetrics.functional.accuracy(
                preds[test_indexes][coverage_filt], targets[test_indexes][coverage_filt]
            ).item()
            accs[f"acc/{tc:.2f}"] = acc

            # class-wise metrics
            num_classes = logits.shape[1]
            for c in range(num_classes):
                c_filt = preds[test_indexes] == c
                if sum(c_filt) == 0:
                    logger.warning("No predictions for class %s", c)
                    coverages[f"cov/{tc:.2f}/{c}"] = 0
                    risks[f"risk/{tc:.2f}/{c}"] = 0
                    accs[f"acc/{tc:.2f}/{c}"] = 0
                    continue
                coverages[f"cov/{tc:.2f}/{c}"] = detect_eval.hard_coverage(scores[test_indexes][c_filt], _thr).item()
                risks[f"risk/{tc:.2f}/{c}"] = (
                    100
                    * detect_eval.selective_net_risk(
                        scores[test_indexes][c_filt], preds[test_indexes][c_filt], targets[test_indexes][c_filt], _thr
                    ).item()
                )
                # accuracy
                coverage_filt = scores[test_indexes][c_filt] <= _thr
                try:
                    acc = torchmetrics.functional.accuracy(
                        preds[test_indexes][c_filt][coverage_filt], targets[test_indexes][c_filt][coverage_filt]
                    ).item()
                except:
                    acc = 0
                accs[f"acc/{tc:.2f}/{c}"] = acc

        # area under risk-coverage curve
        x = torch.tensor([v for k, v in coverages.items() if len(k.split("/")) == 2])
        y = torch.tensor([v for k, v in risks.items() if len(k.split("/")) == 2])
        aurc = torchmetrics.functional.auc(x, y).item()

        # save results
        results_obj = {
            "model": args.model_name,
            "dataset": args.dataset_name,
            "seed": args.seed,
            "score": "confid_net",
            "aurc": aurc,
        }
        print(json.dumps(results_obj, indent=2))
        results_obj.update(**coverages, **risks, **accs)
        helpers.append_results_to_file(results_obj, filename="results/individual_test.csv")

        all_results[args.seed] = results_obj


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
