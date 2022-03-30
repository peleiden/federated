from argparse import ArgumentParser
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from src.models.train_federated import Results


def plot_args(arg: str):
    jobs = os.listdir()
    results = [Results.load(j) for j in jobs if j.isdigit()]
    train_cfgs = [r.cfg.configs.training for r in results]
    values = np.array([cfg[arg] for cfg in train_cfgs])
    perfs = np.array([r.test_accuracies[-1] for r in results])

    argsort = np.argsort(values)
    values = values[argsort]
    perfs = perfs[argsort]

    plt.plot(values, perfs)
    plt.grid()
    plt.xlabel(arg.capitalize())
    plt.ylabel("Final test accuracy [%]")
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
