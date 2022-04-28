from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pelutils.ds.plot import figsize_std, update_rc_params, rc_params, tab_colours

from src.models.train_federated import Results


update_rc_params(rc_params)

def plot_args(type: str):
    var = "local_epochs"
    pretty_var = var.replace("_", " ")

    values = [x[len(type):] for x in os.listdir() if x.startswith(type)]
    values = sorted(values)
    ax = plt.figure(figsize=figsize_std).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, value in enumerate(values):
        jobs = os.listdir(f"{type}{value}")
        results = [Results.load(f"{type}{value}/{j}/0") for j in jobs]

        times = [np.array(result.eval_timestamps)-min(result.eval_timestamps) for result in results]

        min_time = min(t.max() for t in times)
        xvals = np.linspace(0, min_time, 400)
        yvals = np.empty((len(results), len(xvals)))
        for j, res in enumerate(results):
            yvals[j] = np.interp(xvals, np.array(res.eval_timestamps)-min(res.eval_timestamps), res.test_accuracies)

        std = yvals.std(axis=0, ddof=1)
        yvals = yvals.mean(axis=0)
        plt.fill_between(xvals/60, yvals-1.96*std, yvals+1.96*std, alpha=0.2)
        plt.plot(xvals/60, yvals, label=f"{int(value)} {pretty_var}", c=tab_colours[i])

    plt.grid()
    plt.title("Convergence by time and local epochs")
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Test accuracy [%]")
    plt.ylim([-5, 65])
    plt.tight_layout()
    plt.savefig(f"plots/time_avg_{var}.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    args = parser.parse_args()
    os.chdir(args.location)
    os.makedirs("plots", exist_ok=True)
    plot_args("ethernet")
    # plot_args("wifi")
