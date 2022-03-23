from argparse import ArgumentParser
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from src.models.train_federated import Results


def plot_scaling():
    jobs = os.listdir()
    results = [Results.load(j) for j in jobs if j.isdigit()]
    devices = np.array([r.clients_per_round for r in results])
    argsort = np.argsort(devices)
    devices = devices[argsort]
    imgs_per_seconds = np.array([r.num_images/r.train_time for r in results])[argsort]
    performance = np.array([r.test_accuracies[-1] for r in results])[argsort]

    plt.plot(devices, imgs_per_seconds, "-o")
    plt.title("Data rate scaling")
    plt.xlabel("Devices")
    plt.ylabel("Images/s")
    plt.grid()
    plt.ylim(bottom=0)
    plt.tight_layout()

    plt.savefig("plots/speed-scaling.png")
    plt.close()

    plt.plot(devices, 100*imgs_per_seconds/(imgs_per_seconds[0]*devices), "-o")
    plt.title("Efficiency scaling")
    plt.xlabel("Devices")
    plt.ylabel("Efficiency [%]")
    plt.ylim([-7, 107])
    plt.grid()
    plt.tight_layout()

    plt.savefig("plots/effeciency-scaling.png")
    plt.close()

    plt.plot(devices, performance, "-o")
    plt.title("Performance scaling")
    plt.xlabel("Devices")
    plt.ylabel("Test accuracy")
    plt.grid()
    plt.tight_layout()

    plt.savefig("plots/perf-scaling.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    args = parser.parse_args()
    os.chdir(args.location)
    shutil.rmtree("plots", ignore_errors=True)
    os.mkdir("plots")
    plot_scaling()


