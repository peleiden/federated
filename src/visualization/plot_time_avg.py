from argparse import ArgumentParser
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pelutils.ds.plot import figsize_std, update_rc_params, rc_params, tab_colours

from src.models.train_federated import Results


update_rc_params(rc_params)

def plot_args(type: str, var: str):
    pretty_var = var.replace("_", " ")

    values = [x[len(type):] for x in os.listdir() if x.startswith(type)]
    values = sorted(values)
    ax = plt.figure(figsize=figsize_std).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lns = list()
    for i, value in enumerate(values):
        jobs = os.listdir(f"{type}{value}")
        results = [Results.load(f"{type}{value}/{j}/0") for j in jobs]

        times = [np.array(result.eval_timestamps)-min(result.eval_timestamps) for result in results]
        min_time = min(t.max() for t in times)
        xvals = np.linspace(0, min_time, 400)
        yvals = np.empty((len(results), len(xvals)))
        for j, res in enumerate(results):
            yvals[j] = np.interp(xvals, np.array(res.eval_timestamps)-min(res.eval_timestamps), res.test_accuracies)
            # plt.plot(times[j]/60, res.test_accuracies, "-o", alpha=0.3, c=tab_colours[i])

        std = yvals.std(axis=0, ddof=1)
        lower_bound = yvals.min(axis=0)
        upper_bound = yvals.max(axis=0)
        yvals = yvals.mean(axis=0)
        plt.fill_between(xvals/60, lower_bound, upper_bound, alpha=0.2)
        lns.append(plt.plot(xvals/60, yvals, label=f"{int(value)} {pretty_var}", lw=3, c=tab_colours[i]))

    ln = lns[0]
    for l in lns[1:]:
        ln += l
    plt.grid()
    plt.title(f"Varying {pretty_var} on {type}")
    ci_legend = mpatches.Patch(color='gray', alpha=0.3,  label='Approx. 95 % CI')
    # plt.legend(handles=[*ln, ci_legend])
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Test accuracy [%]")
    plt.xlim([-2, 65])
    plt.ylim([-3, 73])
    plt.tight_layout()
    plt.savefig(f"plots/time_avg_{var}_{type}.pdf")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("type", choices=("ethernet", "wifi"))
    args = parser.parse_args()
    os.chdir(args.location)
    os.makedirs("plots", exist_ok=True)
    plot_args(args.type, os.path.split(args.location)[-1])
