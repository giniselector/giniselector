<h1 align="center">
Trusting the Untrustworthy: A Cautionary Tale on the Pitfalls of Training-based Rejection Option
</h1>

<p align="center">
  <img src="main.png" width=512>
</p>
Fig 1. Global coverage (left) refers to the proportion of the entire dataset that is covered by a model's predictions. Low variance in global coverage means that the method consistently covers the target portion of the dataset across different independent runs. However, this consistency in overall coverage may result in a large variance in the coverage for each individual class (right). In other words, the methods may perform well in covering some classes while not performing as well in covering others.


**Submitted to ICML 2023.**

## Requirements

To reproduce the results, you will need to install the packages listed in `requirements.txt` and have `python>=3.7` installed.

Please run `pip install -r requirements.txt` to install them.

## Usage

This section will explain how to reproduce results.

### 1. Training the models from scratch

Simply run the code:

```bash
python -m src.${method}.train \
    --model_name ${model}_${dataset} \
    --seed ${seed}
```

Where the variables you can choose from:

* `method`:
  - `GiniSelector`, which will train the model with the cross-entropy (`CE`) loss.
  - `SelectiveNet`.
  - `ConfidNet`.
  - `DeepGamblers`.
  - `SelfAdaptiveTraining`.

* `model`:
  - `vgg16`
  - `densenet121`
  - `resnet34`

* `dataset`:
  - `cifar-10`
  - `cifar-100`
  - `svhn`

* `seed`: an integer

The training will save the best weights as well as logs, logits, and scores in the `checkpoints/${method}/${model}_${dataset}/${seed}` folder.

#### Or you can download the checkpoints

Abiding by the principles of open science, we provide the checkpoints for most methods and models on the [release page](https://github.com/giniselector/giniselector/releases/tag/checkpoints). PS.: The checkpoints for `SelectiveNet` are not available due to the size of the files (~30GB in total).

Run the following code to download and unzip a checkpoint from the release:

```bash
mkdir checkpoints/
cd checkpoints/
wget -c https://github.com/giniselector/giniselector/releases/download/checkpoints/${method}-${model}_${dataset}.zip
unzip ./${method}-${model}_${dataset}.zip
```

Or go to the [release page](https://github.com/giniselector/giniselector/releases/tag/checkpoints) and download the checkpoints manually.

Then you can save the logits and scores from a specific method and model by running:

```bash
python -m src.${method}.test \
    --model_name ${model}_${dataset} \
    --seed ${seed}
```

### 2. Evaluating

To evaluate the benchmark, please run:

```bash
python -m src.analysis.save_results \
    --model_name ${model}_${dataset}
```

The results will be exported to `.csv` in the `results/` repository, and the evaluation metrics are defined in `src/utils/eval.py`

### 3. Plotting

Run the code:

```bash
python -m src.analysis.plots \
    --model_name ${model}_${dataset}
```

The images will be saved to the `images/` folder.

## Our Method

The post-hoc `Gini` score can be plugged into any deep classifier trained with the cross-entropy loss. The function implemented in python that takes the logits as input is defined below.

```python
import torch


def gini(logits: torch.Tensor):
    g = torch.sum(torch.softmax(logits, 1) ** 2, 1)
    return 1 - g
```

## Results

<p align="center">
  <img src="table1.png" width=512>
</p>

### (Optional) Environmental Variables

Please, place the following lines in your `.env` file if you want to modify any of the default folders.

```bash
#.env
export DATA_DIR=""
export IMAGENET_ROOT=""
export CHECKPOINTS_DIR="checkpoints/"
```

Where:

- `DATA_DIR` is the directory where the datasets will be downloaded.
- `IMAGENET_ROOT` is the directory where the ImageNet dataset is located.
- `CHECKPOINTS_DIR` is the directory where the pre-trained models will be placed.
