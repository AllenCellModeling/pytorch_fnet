import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(paths_logs, history):
    kwargs_fig = {
        'dpi': 100,
    }
    kwargs_plot = {
        'linewidth': 0.5,
    }
    fig, ax = plt.subplots(**kwargs_fig)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss (MSE)')
    for path in paths_logs:
        df = pd.read_csv(path)
        ax.plot(df['num_iter'][history:], df['loss'][history:], **kwargs_plot)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--paths_logs', nargs='+', help='path(s) to log of model training')
    parser.add_argument('--history', type=int, default=0, help='iteration to start displaying plot')
    opts = parser.parse_args()
    
    plot_loss(opts.paths_logs, opts.history)
