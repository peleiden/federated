from __future__ import annotations
from argparse import ArgumentParser
import os
import shutil

from pelutils import log
from pelutils.ds.plot import figsize_std, update_rc_params, rc_params
import matplotlib.pyplot as plt
import numpy as np

from src.models.train_federated import Results

update_rc_params(rc_params)


def plot_accuracy():
    log.section("Plotting accuracy")
    res = Results.load()
    train_cfg = res.cfg.configs.training

    local_epochs = train_cfg.local_epochs
    comm_rounds = train_cfg.communication_rounds
    comm_rounds = np.arange(1, comm_rounds+1)

    plt.figure(figsize=figsize_std)
    try:
        for i in comm_rounds-1:
            for j in range(res.clients_per_round):
                if i:
                    x = i + np.arange(local_epochs+1) / local_epochs
                    y = [res.test_accuracies[i-1], *res.local_accs[i][j]]
                else:
                    x = i+np.arange(1, local_epochs+1)/local_epochs
                    y = res.local_accs[i][j]
                plt.plot(x, y, alpha=0.5, label="Local test accuracy" if not (i or j) else None, color="gray")
    except ValueError:
        pass

    if any(x < 100 for x in res.pct_noisy_images_by_round):
        plt.plot(comm_rounds, res.pct_noisy_images_by_round, "-o", label="% noisy images")

    plt.plot(
        comm_rounds,
        res.test_accuracies,
        "-o",
        label="Global test accuracy",
    )
    plt.title("Accuracy")
    plt.xlabel("Communication rounds")
    plt.ylabel("Accuracy [%]")
    plt.ylim([-5, 105])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/accuracy.png")
    plt.close()

def plot_memory():
    log.section("Plotting memory")
    res = Results.load()

    plt.figure(figsize=figsize_std)
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
