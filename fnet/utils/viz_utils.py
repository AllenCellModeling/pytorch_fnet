"""Visualization tools."""


from typing import List, Optional, Union
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)
plt.style.use("seaborn")
COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_df(df, ax, model_label, colors, **kwargs):
    """Plot dataframe columns on axes."""
    for idx_c, col in enumerate(df.columns):
        label = (f"{model_label}:" if model_label is not None else "") + f"{col}"
        key = model_label, "_".join(col.split("_")[:-1])
        if key not in colors:
            colors[key] = COLORS[colors["idx"]]
            colors["idx"] = (colors["idx"] + 1) % len(COLORS)
        color = colors[key]
        ax.plot(
            df.index.to_numpy(), df[col].to_numpy(), color=color, label=label, **kwargs
        )


def plot_loss(
    paths_model: Union[List[str], str],
    path_save: Optional[str] = None,
    train: bool = True,
    val: bool = True,
    title: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    """Plots model loss curve(s).

    Parameters
    ----------
    paths_model
        List of paths to model directories specified as a list or as a string
        of paths separated by spaces.
    path_save
        If not None, specifies where to save figure and figure will not be
        displayed.
    train
        Set to plot training curve.
    val
        Set to plot validation curve.
    title
        Plot title.
    ymin
        Y-axis minimum value.
    ymax
        Y-axis maximum value.

    """
    if isinstance(paths_model, str):
        paths_model = paths_model.split(" ")
    if path_save is not None:
        plt.switch_backend("Agg")
    window_train = 128
    window_val = 32
    colors = {"idx": 0}  # maps model-content to colors; idx is COLORS index
    fig, ax = plt.subplots()
    for idx_m, path_model in enumerate(paths_model):
        name_model = os.path.basename(os.path.normpath(path_model))
        model_label = None if len(paths_model) == 1 else name_model
        path_loss = os.path.join(path_model, "losses.csv")
        df = pd.read_csv(path_loss, index_col="num_iter")
        if train:
            cols_train = [col for col in df.columns if col.lower().endswith("_train")]
            df_train = df.loc[:, cols_train].dropna(axis=1, thresh=1).dropna()
            df_train_rmean = df_train.rolling(window=window_train).mean()
            _plot_df(df_train_rmean, ax, model_label, colors, linestyle="-")
        if val:
            cols_val = [col for col in df.columns if col.lower().endswith("_val")]
            df_val = df.loc[:, cols_val].dropna(axis=1, thresh=1).dropna()
            df_val_rmean = df_val.rolling(window=window_val).mean()
            _plot_df(df_val_rmean, ax, model_label, colors, linestyle="--")
    if title is not None:
        ax.set_title(title)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Rolling mean squared error")
    ax.legend()
    if path_save is not None:
        fig.savefig(path_save, bbox_inches="tight")
        logger.info(f"Saved: {path_save}")
        return
    plt.show()


def plot_metric(
    path_csv: str,
    metric: str,
    path_save: Optional[str] = None,
    title: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    """Plots box-plot of model performance according to some metric.

    Parameters
    ----------
    path_csv
        Path to csv where each row is a dataset item.
    metric
        Name of metric. Should be within one or more CSV column names.
    path_save
        If not None, specifies where to save figure and figure will not be
        displayed.
    title
        Plot title.
    ymin
        Y-axis minimum value.
    ymax
        Y-axis maximum value.

    """
    if path_save is not None:
        plt.switch_backend("Agg")
    df = pd.read_csv(path_csv)
    cols = [c for c in df.columns if metric in c]
    cols_rename = {c: c.split(metric)[-1] for c in cols}
    df = df.loc[:, cols].rename(columns=cols_rename)
    fig, ax = plt.subplots()
    df.boxplot(ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Pearson correlation coefficient (r)")
    if path_save is not None:
        fig.savefig(path_save, bbox_inches="tight")
        logger.info(f"Saved: {path_save}")
        return
    plt.show()
