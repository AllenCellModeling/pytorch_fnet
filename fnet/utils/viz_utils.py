from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd


plt.switch_backend('TkAgg')
plt.style.use('seaborn')
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def _plot_df(df, ax, **kwargs):
    """Plot dataframe columns on axes."""
    for idx, col in enumerate(df.columns):
        ax.plot(df.index, df[col], color=COLORS[idx], label=col, **kwargs)


def plot_loss(path_model: Optional[str] = None) -> None:
    window_train = 128
    window_val = 32
    if path_model is None:
        path_model = os.getcwd()
    print('Model:', path_model)
    path_loss = os.path.join(path_model, 'losses.csv')
    df = pd.read_csv(path_loss, index_col='num_iter')
    cols_train = [col for col in df.columns if col.lower().endswith('_train')]
    cols_val = [col for col in df.columns if col.lower().endswith('_val')]
    df_train = df.loc[:, cols_train].dropna()
    df_val = df.loc[:, cols_val].dropna()
    df_train_rmean = df_train.rolling(window=window_train).mean()
    df_val_rmean = df_val.rolling(window=window_val).mean()
    fig, ax = plt.subplots()
    _plot_df(df_train_rmean, ax, linestyle='-')
    _plot_df(df_val_rmean, ax, linestyle='--')
    ax.set_xlabel('Training iterations')
    ax.set_ylabel('Rolling mean squared error')
    ax.legend()
    plt.show()
