import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', help='path to log of model training')
parser.add_argument('--history', type=int, default=100, help='iteration to start displaying plot')
opts = parser.parse_args()

def plot_loss():
    df = pd.read_csv(opts.log_path)
    history = opts.history
    plt.figure(figsize=(12, 8))
    plt.plot(df['num_iter'][history:], df['loss'][history:])
    plt.xlabel('iteration')
    plt.ylabel('loss (MSE)')
    plt.show()

def main():
    plot_loss()

if __name__ == '__main__':
    plot_loss()
