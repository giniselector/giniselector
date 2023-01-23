import os
from typing import Optional

import torch
import torchmetrics
from torch import Tensor

import src.utils.config as config
import src.utils.eval as detect_eval


def get_checkpoint_path(path_prefix: str, model_name: str, seed: str, tc: Optional[str]) -> str:
    tc = tc if tc is not None else ""
    return os.path.join(config.CHECKPOINTS, path_prefix, model_name, seed, tc if path_prefix == "selective_net" else "")


def find_thr(logits: Tensor, labels: Tensor, scores: Tensor, coverage: float) -> float:
    pred = torch.argmax(logits, dim=1)
    risks, covs, thrs = detect_eval.risks_coverages_selective_net(scores, pred, labels)
    thr = thrs[min(torch.searchsorted(covs, coverage).item(), len(thrs) - 1)].item()
    return thr


def find_coverage(pred: Tensor, labels: Tensor, scores: Tensor, thr: float) -> float:
    risks, covs, thrs = detect_eval.risks_coverages_selective_net(scores, pred, labels)
    cov = covs[min(torch.searchsorted(thrs, thr).item(), len(covs) - 1)].item()
    return cov


def find_risk(pred: Tensor, labels: Tensor, scores: Tensor, thr: float) -> float:
    risks, covs, thrs = detect_eval.risks_coverages_selective_net(scores, pred, labels)
    risk = risks[min(torch.searchsorted(thrs, thr).item(), len(risks) - 1)].item()
    return risk


def global_analysis(pred: Tensor, labels: Tensor, scores: Tensor, thr: float):
    # compute accuracy
    acc = torchmetrics.functional.accuracy(labels, pred).item()

    # compute coverage
    cov = find_coverage(pred, labels, scores, thr)

    # compute risk
    risk = find_risk(pred, labels, scores, thr)

    return {"acc": acc, "cov": cov, "risk": risk}


def per_class_analysis(pred: Tensor, labels: Tensor, scores: Tensor, thr: float, class_id: int) -> dict:
    # filter per class
    filt = labels == class_id
    # if filt.sum() == 0:
    #     return {"acc": 0, "cov": 0, "risk": 0}
    pred = pred[filt]
    labels = labels[filt]
    scores = scores[filt]

    # compute accuracy
    acc = torchmetrics.functional.accuracy(labels, pred).item()

    # compute coverage
    cov = find_coverage(pred, labels, scores, thr)

    # compute risk
    risk = find_risk(pred, labels, scores, thr)

    return {"acc": acc, "cov": cov, "risk": risk}


pretty_models = {"densenet121": "DenseNet-121", "resnet34": "ResNet-34", "vgg16": "VGG-16"}
pretty_datasets = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "svhn": "SVHN"}
pretty_methods = {
    "gini": "Gini",
    "selective_net": "SelectiveNet",
    "confid_net": "ConfidNet",
    "deep_gamblers": "DeepGamblers",
    "sat": "SelfAdaptiveTraining",
}
cifar10_classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
svhn_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
cifar100_classes = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "cra",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

classes = {"cifar10": cifar10_classes, "cifar100": cifar100_classes, "svhn": svhn_classes}
