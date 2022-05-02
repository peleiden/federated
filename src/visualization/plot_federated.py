from __future__ import annotations
from argparse import ArgumentParser
import os
import shutil

from matplotlib.ticker import MaxNLocator
from pelutils import log
from pelutils.ds.plot import figsize_std, figsize_wide, update_rc_params, rc_params, tab_colours
import matplotlib.pyplot as plt
import numpy as np

from src.models.train_federated import Results

update_rc_params(rc_params)


def plot_accuracy():
    log.section("Plotting accuracy")
    res = Results.load()
    train_cfg = res.cfg.configs.training

    local_epochs = train_cfg.local_epochs
    comm_rounds = np.arange(len(res.test_accuracies))
    SMALL_SIZE = 32
    MEDIUM_SIZE = 36
    BIGGER_SIZE = 40

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax = plt.figure(figsize=figsize_std).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    try:
        for i in comm_rounds[:-1]:
            for j in range(res.clients_per_round):
                x = i + np.arange(local_epochs+1) / local_epochs
                y = [res.test_accuracies[i], *res.local_accs[i][j]]
                plt.plot(x, y, alpha=0.3, label="Local test accuracy" if not (i or j) else None, color="tab:green")
    except ValueError:
        pass

    for i in comm_rounds[:-1]:
        for j in range(res.clients_per_round):
            x = i + np.arange(local_epochs+1) / local_epochs
            y = [100*x for x in res.local_train_accs[i][j]]
            plt.plot(x[1:], y, alpha=0.3, label="Local training accuracy" if not (i or j) else None, color="tab:orange")

    plt.plot(
        comm_rounds,
        res.test_accuracies,
        "-",
        label="Global test accuracy",
        color=tab_colours[0],
        lw=3,
        ms=4
    )
    if any(x > 100 for x in res.pct_noisy_images_by_round):
        plt.plot(comm_rounds[1:], res.pct_noisy_images_by_round, "-o", label="% noisy images")



    plt.title("Local, Global Model Accuracies")
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy [%]")
    plt.ylim([10, 115])
    plt.xticks(np.arange(21, step=5))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/accuracy.pdf")
    plt.close()

def plot_memory():
    log.section("Plotting memory")
    res = Results.load()

    plt.figure(figsize=figsize_wide)
    plt.title("Memory usage")
    start_time = res.telemetry[0]["timestamp"][0]
    if res.ip:
        for tel in res.telemetry:
            plt.plot(
                np.array(tel["timestamp"])-start_time,
                tel["memory_usage"],
                c="gray",
                alpha=0.7,
            )

    plt.xlabel("Time [s]")
    plt.ylabel("Total memory usage [%]")
    plt.ylim([-5, 105])
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/memory.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("working_directory")
    args = parser.parse_args()
    os.chdir(args.working_directory)
    os.makedirs("plots", exist_ok=True)

    log.configure("plots.log")
    with log.log_errors:
        plot_accuracy()
        try:
            plot_memory()
        except IndexError:
            log("Memory info not found")
