from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pelutils.ds.plot import figsize_std, update_rc_params, rc_params

from src.models.train_federated import Results


update_rc_params(rc_params)

def plot_args():
    jobs = os.listdir()
    results = [Results.load(f"{j}/0") for j in jobs if j.isdigit()]
    train_cfgs = [r.cfg.configs.training for r in results]
    local_epochs = [cfg["local_epochs"] for cfg in train_cfgs]
    argsort = np.argsort(local_epochs)
    results = [results[i] for i in argsort]
    train_cfgs = [train_cfgs[i] for i in argsort]

    ax = plt.figure(figsize=figsize_std).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for result, cfg in zip(results, train_cfgs):
        eval_times = np.array(result.eval_timestamps) - result.eval_timestamps[0]
        plt.plot(eval_times/60, result.test_accuracies, "-o", label="%i local epochs" % cfg["local_epochs"])
    plt.grid()
    plt.title("Convergence by time and local epochs")
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Test accuracy [%]")
    plt.ylim([-5, 65])
    plt.tight_layout()
    plt.savefig("plots/time.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    args = parser.parse_args()
    os.chdir(args.location)
    os.makedirs("plots", exist_ok=True)
    plot_args()
