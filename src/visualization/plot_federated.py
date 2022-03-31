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


def plot_timeline():
    log.section("Plotting timeline")
    res = Results.load()

    plt.figure(figsize=figsize_std)
    plt.title("Training on %i devices" % len(res.telemetry))
    start_time = res.telemetry[0]["timestamp"][0]
    log("Plotting accuracy")
    plt.plot(
        np.array(res.eval_timestamps)-start_time,
        res.test_accuracies,
        "-o",
        label="Test accuracy",
    )
    if res.ip:
        log("Plotting memory usage")
        for i, tel in enumerate(res.telemetry):
            plt.plot(
                np.array(tel["timestamp"])-start_time,
                tel["memory_usage"],
                c="gray",
                alpha=0.5,
                label="Memory usage" if i == 0 else None,
            )

    plt.xlabel("Time [s]")
    plt.ylabel("%")
    plt.ylim([-5, 105])
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/timeline.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("working_directory")
    args = parser.parse_args()
    os.chdir(args.working_directory)
    os.makedirs("plots", exist_ok=True)

    log.configure("plots.log")
    with log.log_errors:
        plot_timeline()
