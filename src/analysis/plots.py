import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import config

from . import utils as ut

sns.set_context("paper")
sns.set_palette("colorblind")
DPI = 1000


def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return inner_function


def get_df(model: str, dataset: str):
    return pd.read_csv(os.path.join(config.RESULTS, f"{model}_{dataset}/all_metrics.csv"))


@exception_handler
def global_plots(model: str, dataset: str):
    df = get_df(model, dataset)
    df["method"] = df["method"].apply(lambda x: ut.pretty_methods[x])
    df["risk"] = df["risk"].apply(lambda x: x * 100)
    df = df.query("class_id == -1")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 2.5), dpi=120)
    ax[0].plot([0.5, 1], [0.5, 1], color="black", linestyle="--", linewidth=1, label="Perfect Calibration")
    sns.lineplot(
        ax=ax[1],
        data=df,
        x="coverage",
        y="risk",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        estimator="median",
    )
    # 95% confidence interval
    sns.lineplot(ax=ax[0], data=df, x="coverage", y="cov", hue="method", style="method", err_style="bars", markers=True)
    ax[0].legend().remove()
    ax[1].legend().remove()
    ax[0].set_xlabel("Target Coverage")
    ax[1].set_xlabel("Target Coverage")
    ax[0].set_ylabel("Coverage")
    ax[1].set_ylabel("Risk")

    # legend on top
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=True)

    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(
        os.path.join(path, "global_plots.pdf"),
        dpi=DPI,
        bbox_inches="tight",
    )


@exception_handler
def main_figure(model: str, dataset: str):
    df = get_df(model, dataset)
    df["method"] = df["method"].apply(lambda x: ut.pretty_methods[x])
    df = df.query("method!='Gini'")
    df["risk"] *= 100

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(4.5, 5), dpi=120)
    fig.suptitle("Coverage", fontsize=10)
    # fig.supxlabel("Coverage", fontsize=10)
    palette = sns.color_palette(
        ["#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
    )
    query_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    sns.boxplot(
        ax=ax[0],
        data=df.query(f"coverage==@query_list and class_id==-1"),
        x="cov",
        y="coverage",
        hue="method",
        orient="h",
        palette=palette,
    )
    ax[0].yaxis.tick_right()
    ax[0].xaxis.tick_top()
    ax[0].set_axisbelow(True)
    ax[0].grid(which="major", axis="x", linestyle="--")
    ax[0].legend([], [], frameon=False)
    ax[0].tick_params(direction="out", length=0.2, color="white")
    ax[0].set_ylabel("Target Coverage")
    ax[0].set_xlabel("")

    sns.boxplot(
        ax=ax[1],
        data=df.query(f"coverage==@query_list and class_id!=-1"),
        x="cov",
        y="coverage",
        hue="method",
        orient="h",
        palette=palette,
    )
    ax[1].set_axisbelow(True)
    ax[1].xaxis.tick_top()
    ax[1].grid(which="major", axis="x", linestyle="--")
    ax[1].legend([], [], frameon=False)
    ax[1].tick_params(direction="out", length=0, width=0)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    # ax[0].set_yticks(query_list)
    ax[0].set_yticklabels(query_list)
    ax[0].set_xticks(query_list)
    ax[0].set_xticklabels(query_list)
    ax[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax[0].set_title("Global")
    ax[1].set_title("Class-Wise")

    # legend below the plot
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    for i, spine in enumerate(ax[0].spines.values()):
        if i in [0, 2]:
            spine.set_visible(False)

    for i, spine in enumerate(ax[1].spines.values()):
        if i in [1, 2]:
            spine.set_visible(False)

    # save the figure
    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "fig1.pdf"), dpi=DPI, bbox_inches="tight")


@exception_handler
def accuracies(model: str, dataset: str):
    df = get_df(model, dataset)
    this_pretty_methods = ut.pretty_methods.copy()
    this_pretty_methods["gini"] = "CE"
    df["method"] = df["method"].apply(lambda x: this_pretty_methods[x])
    df["risk"] *= 100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2), dpi=120)
    sns.boxplot(
        x="class_id",
        y="acc",
        hue="method",
        data=df.query("coverage == 1.0"),
        showmeans=False,
    )
    # legend on top
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=False)
    cs = ["All"] + ut.classes[dataset]
    plt.xticks(range(0, 11), labels=cs)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--")

    # rotate xticks
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "accuracies.pdf"), dpi=DPI, bbox_inches="tight")


@exception_handler
def cov_and_risk_plots(model: str, dataset: str):
    df = get_df(model, dataset)
    df["method"] = df["method"].apply(lambda x: ut.pretty_methods[x])
    df["risk"] = df["risk"].apply(lambda x: x * 100)
    df["coverage"] = df["coverage"].apply(lambda x: f"{x:.2f}")
    df = df.query(f"class_id != -1")

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(11, 3.5), dpi=120)
    sns.boxplot(ax=ax[0], data=df, x="method", y="cov", hue="coverage", showmeans=True)
    sns.boxplot(ax=ax[1], data=df, x="method", y="risk", hue="coverage", showmeans=True)
    # legend below the plot
    ax[0].set_axisbelow(True)
    ax[0].grid(which="major", axis="y", linestyle="--")
    ax[1].set_axisbelow(True)
    ax[1].grid(which="major", axis="y", linestyle="--")
    ax[0].legend().remove()
    ax[1].legend(title="Target\nCoverage", loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(1.065, 0))
    ax[0].set_ylabel("Coverage")
    ax[1].set_ylabel("Risk")
    ax[0].set_xlabel("")
    ax[0].tick_params(direction="out", length=0, color="white")
    ax[1].set_xlabel("")
    ax[1].set_yscale("log")

    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "cov_and_risk.pdf"), dpi=DPI, bbox_inches="tight")


@exception_handler
def simple_risk_plot(model: str, dataset: str):
    df = get_df(model, dataset)
    df["method"] = df["method"].apply(lambda x: ut.pretty_methods[x])
    df["risk"] = df["risk"].apply(lambda x: x * 100)
    df = df.query("class_id == -1")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5), dpi=120)
    sns.lineplot(
        ax=ax,
        data=df,
        x="coverage",
        y="risk",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        estimator="median",
    )
    ax.set_xlabel("Target Coverage")
    ax.set_ylabel("Risk")
    plt.legend(title="")

    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "risk.pdf"), dpi=DPI, bbox_inches="tight")


@exception_handler
def simple_coverage_plot(model: str, dataset: str):
    df = get_df(model, dataset)
    df["method"] = df["method"].apply(lambda x: ut.pretty_methods[x])
    df["risk"] = df["risk"].apply(lambda x: x * 100)
    df = df.query("class_id == -1")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5), dpi=120)
    ax.plot([0.5, 1], [0.5, 1], color="black", linestyle="--", linewidth=1, label="Perfect Calibration")

    sns.lineplot(ax=ax, data=df, x="coverage", y="cov", hue="method", style="method", err_style="bars", markers=True)
    ax.set_xlabel("Target Coverage")
    ax.set_ylabel("Coverage")
    plt.legend(title="")

    path = os.path.join(config.IMAGES, f"{model}_{dataset}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "coverage.pdf"), dpi=DPI, bbox_inches="tight")


def plot_all(args):
    global_plots(args.model, args.dataset)
    main_figure(args.model, args.dataset)
    accuracies(args.model, args.dataset)
    cov_and_risk_plots(args.model, args.dataset)
    simple_risk_plot(args.model, args.dataset)
    simple_coverage_plot(args.model, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--dataset", type=str, default="cifar10")
    args = parser.parse_args()
    plot_all(args)
