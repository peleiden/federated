from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
from pelutils.ds.plot import figsize_std, update_rc_params, rc_params

from src.models.train_federated import Results


update_rc_params(rc_params)


def plot_args(arg: str):
    jobs = os.listdir()
    results = [Results.load(j) for j in jobs if j.isdigit()]
    train_cfgs = [r.cfg.configs.training for r in results]
    values = np.array([cfg[arg] for cfg in train_cfgs])
    perfs = np.array([r.test_accuracies[-1] for r in results])

    argsort = np.argsort(values)
    values = values[argsort]
    perfs = perfs[argsort]

    unique_values = np.unique(values)
    means = np.zeros(unique_values.size)
    for i, unique_value in enumerate(unique_values):
        means[i] = perfs[values==unique_value].mean()

    plt.figure(figsize=figsize_std)
    plt.scatter(values, perfs, label="Measurements")
    plt.plot(unique_values, means, label="Mean")
    plt.grid()
    plt.legend()
    plt.xlabel(arg.capitalize().replace("_", " "))
    plt.ylabel("Final test accuracy [%]")
    plt.ylim([-5, 105])
    plt.tight_layout()
    plt.savefig("plots/devices-%s.png" % arg)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("--arg")
    args = parser.parse_args()
    os.chdir(args.location)
    os.makedirs("plots", exist_ok=True)
    plot_args(args.arg)
