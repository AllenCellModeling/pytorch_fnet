from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd


plt.switch_backend('TkAgg')
plt.style.use('seaborn')
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def _plot_df(df, ax, model_label, colors, **kwargs):
    """Plot dataframe columns on axes."""
    for idx_c, col in enumerate(df.columns):
        label = (f'{model_label}:'if model_label is not None else '') + f'{col}'
        key = model_label, '_'.join(col.split('_')[:-1])
        if key not in colors:
            colors[key] = COLORS[colors['idx']]
            colors['idx'] = (colors['idx'] + 1) % len(COLORS)
        color = colors[key]
        ax.plot(df.index, df[col], color=color, label=label, **kwargs)


def plot_loss(
        *paths_model: str,
        save_fig: bool = False,
) -> None:
    """Plots model loss curve(s).

    Parameters
    ----------
    *paths_model
        Variable length list of paths to saved model directories.
    save_fig
        Set to save the figure inside model directory rather than display.

    """
    if save_fig:
        plt.switch_backend('Agg')
    window_train = 128
    window_val = 32
    colors = {'idx': 0}  # maps model-content to colors; idx is COLORS index
    if len(paths_model) == 0:
        paths_model = (os.getcwd(), )
    fig, ax = plt.subplots()
    for idx_m, path_model in enumerate(paths_model):
        print(f'Model {idx_m}:', path_model)
        path_loss = os.path.join(path_model, 'losses.csv')
        df = pd.read_csv(path_loss, index_col='num_iter')
        cols_train = [
            col for col in df.columns if col.lower().endswith('_train')
        ]
        cols_val = [col for col in df.columns if col.lower().endswith('_val')]
        df_train = df.loc[:, cols_train].dropna(axis=1, thresh=1).dropna()
        df_val = df.loc[:, cols_val].dropna(axis=1, thresh=1).dropna()
        df_train_rmean = df_train.rolling(window=window_train).mean()
        df_val_rmean = df_val.rolling(window=window_val).mean()
        model_label = None if len(paths_model) == 1 else idx_m
        _plot_df(df_train_rmean, ax, model_label, colors, linestyle='-')
        _plot_df(df_val_rmean, ax, model_label, colors, linestyle='--')
    ax.set_xlabel('Training iterations')
    ax.set_ylabel('Rolling mean squared error')
    ax.legend()
    if save_fig:
        path_save = os.path.join(path_model, 'loss_curves.png')
        fig.savefig(path_save, bbox_inches='tight')
        print('Saved:', path_save)
        return
    plt.show()
    
