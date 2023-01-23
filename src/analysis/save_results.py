import argparse
import os

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import torch
from tqdm import tqdm

from src.analysis.utils import find_thr, get_checkpoint_path, global_analysis, per_class_analysis
from src.utils import config


def main(model_name: str, seed: str, coverage: float):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load data
    target_coverage = f"{coverage:.2f}"
    keys = [
        "gini",
        "selective_net",
        "confid_net",
        "deep_gamblers",
        "sat",
    ]
    path_prefix = {
        "msp": "ce",
        "gini": "ce",
        "confid_net": "confid_net",
        "selective_net": "selective_net",
        "deep_gamblers": "deep_gamblers",
        "sat": "sat",
    }
    test_logits = {
        k: torch.load(
            os.path.join(get_checkpoint_path(path_prefix[k], model_name, seed, target_coverage), "test_logits.pt")
        )
        for k in keys
    }
    test_labels = {
        k: torch.load(
            os.path.join(get_checkpoint_path(path_prefix[k], model_name, seed, target_coverage), "test_targets.pt")
        )
        for k in keys
    }
    test_scores = {
        k: torch.load(
            os.path.join(get_checkpoint_path(path_prefix[k], model_name, seed, target_coverage), "test_scores.pt")
        ).squeeze()
        for k in keys
    }

    # check shapes
    for k in keys:
        assert test_logits[k].shape[0] == test_scores[k].shape[0] == test_labels[k].shape[0], "Shapes do not match"

    # split data into calibration and test
    cal_indexes, test_indexes = sklearn.model_selection.train_test_split(
        np.arange(len(test_logits[keys[0]])), test_size=0.9, random_state=int(seed)
    )
    val_logits = {k: test_logits[k][cal_indexes] for k in test_logits}
    val_labels = {k: test_labels[k][cal_indexes] for k in test_labels}
    val_scores = {k: test_scores[k][cal_indexes] for k in test_scores}
    test_logits = {k: test_logits[k][test_indexes] for k in test_logits}
    test_labels = {k: test_labels[k][test_indexes] for k in test_labels}
    test_scores = {k: test_scores[k][test_indexes] for k in test_scores}
    test_pred = {k: torch.argmax(test_logits[k], dim=1) for k in keys}

    # validation: find thr for each method fixed coverage
    thrs = {k: find_thr(val_logits[k], val_labels[k], val_scores[k], coverage) for k in keys}

    df = pd.DataFrame()
    # global analysis
    for k in keys:
        global_metrics = global_analysis(
            test_pred[k].to(device), test_labels[k].to(device), test_scores[k].to(device), thrs[k]
        )
        tmp = pd.DataFrame(
            {
                "class_id": [-1],
                "method": [k],
                "acc": [global_metrics["acc"]],
                "cov": [global_metrics["cov"]],
                "risk": [global_metrics["risk"]],
                "coverage": [round(coverage, 2)],
                "seed": [seed],
            }
        )
        df = pd.concat([df, tmp], axis=0)

    # per class analysis
    for k in keys:
        for class_id in range(test_logits[k].shape[1]):
            per_class_metrics = per_class_analysis(
                test_pred[k].to(device), test_labels[k].to(device), test_scores[k].to(device), thrs[k], class_id
            )
            tmp = pd.DataFrame(
                {
                    "class_id": [class_id],
                    "method": [k],
                    "acc": [per_class_metrics["acc"]],
                    "cov": [per_class_metrics["cov"]],
                    "risk": [per_class_metrics["risk"]],
                    "coverage": [round(coverage, 2)],
                    "seed": [seed],
                }
            )
            df = pd.concat([df, tmp], axis=0)

    return df


def save_all(args):
    model_name = f"{args.model}_{args.dataset}"

    all_metircs = pd.DataFrame()
    for seed in tqdm(range(1, 11), "Seed"):
        for coverage in tqdm(np.linspace(0.5, 1, 11), "Coverage"):
            try:
                all_metircs = pd.concat([all_metircs, main(model_name, str(seed), coverage)])
            except Exception as exc:
                print(f"Failed for {model_name} {seed} {coverage}")
                print(exc)

    # save results
    path = os.path.join(config.RESULTS, model_name)
    os.makedirs(path, exist_ok=True)
    all_metircs.to_csv(os.path.join(path, "all_metrics.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vgg16_cifar10")
    args = parser.parse_args()
    args.model, args.dataset = args.model_name.split("_")
    save_all(args)
