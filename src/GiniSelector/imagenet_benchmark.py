import os

import numpy as np
import sklearn
import sklearn.model_selection
import torch
import torch.utils.data
import torchmetrics.functional as metrics
import torchvision
from tqdm import tqdm

import src.utils.eval as detect_eval
from src.utils import helpers

from .scores import gini

IMAGENET_ROOT = os.environ["IMAGENET_ROOT"]


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    models = [
        torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT),
        torchvision.models.resnet34(weights=torchvision.models.resnet.ResNet34_Weights.DEFAULT),
        torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT),
        torchvision.models.resnet101(weights=torchvision.models.resnet.ResNet101_Weights.DEFAULT),
        torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights.DEFAULT),
    ]

    accuracies = [69.758, 73.314, 80.858, 81.886, 82.284]

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = torchvision.datasets.ImageNet(IMAGENET_ROOT, split="val", transform=test_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=8)

    calculated_accuracy = []
    for i, model in enumerate(tqdm(models)):
        model.eval()
        model.to(device)
        test_pred = torch.zeros(len(dataset), dtype=torch.long)
        scores = torch.zeros(len(dataset), dtype=torch.float32)
        test_logits = torch.zeros(len(dataset), 1000, dtype=torch.float32)
        test_labels = torch.zeros(len(dataset), dtype=torch.long)
        acc = 0
        total = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            test_labels[total: total + len(labels)] = labels.cpu()
            test_logits[total: total + len(labels)] = outputs.cpu()
            test_pred[total: total + len(labels)] = preds.cpu()
            scores[total: total + len(labels)] = gini(outputs).cpu()

            acc += torch.sum(preds == labels).item()
            total += labels.size(0)

        calculated_accuracy.append(acc / total)
        print(f"Accuracy: {acc*100 / total} vs. {accuracies[i]}")

        for target_coverage in np.linspace(0.5, 1, 11):
            # calibration
            cal_indexes, test_indexes = sklearn.model_selection.train_test_split(
                np.arange(len(scores)), test_size=0.9, random_state=1
            )

            cal_risks, cal_coverages, cal_thrs = detect_eval.risks_coverages_selective_net(
                scores[cal_indexes], test_pred[cal_indexes], test_labels[cal_indexes]
            )

            _thr = cal_thrs[torch.searchsorted(cal_coverages, target_coverage)].item()
            test_sn_coverages = detect_eval.hard_coverage(scores[test_indexes], _thr).item()

            test_sn_risks = (
                100
                * detect_eval.selective_net_risk(
                    scores[test_indexes],
                    test_pred[test_indexes],
                    test_labels[test_indexes],
                    _thr,
                ).item()
            )

            # accuracy
            coverage_idx = scores[test_indexes] <= _thr
            composite_test_acc = metrics.accuracy(
                test_pred[test_indexes][coverage_idx],
                test_labels[test_indexes][coverage_idx],
            ).item()

            results_obj = {
                "model": model_names[i],
                "coverage": target_coverage,
                "acc": calculated_accuracy[-1],
                "cov": test_sn_coverages,
                "risk": test_sn_risks,
                "composite_acc": composite_test_acc,
            }

            # save results
            helpers.append_results_to_file(results_obj, filename="results/imagenet_benchmark.csv")


if __name__ == "__main__":
    main()
