# %% [markdown]
# This is a Jupytext notebook, use e.g. [the vscode extension](https://marketplace.visualstudio.com/items?itemName=donjayamanne.vscode-jupytext) and open right-click + open as Jupyter Notebook on the file

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pelutils.ds.plot import figsize_std, rc_params, tab_colours
rcParams.update(rc_params)

# %%
clients = [1, 2, 5, 10]
final_accuracies = np.array(
[
    [49.19, 50.66, 52.18, 52.44],
    [47.91, 50.33, 51.86, 53.65],
    [47.54, 48.87, 52.08, 53.18],
]
)
fig, ax = plt.subplots(figsize=figsize_std)

ax.set_title("CIFAR10: Scaling over clients sampled")
ax.set_ylabel("Accuracy on 10K test images [%] over 3 runs")
ax.set_xlabel("Number of clients sampled each round out of 10")
ax.plot(clients, final_accuracies.mean(0),  "o-", color=tab_colours[1], lw=2, ms=10, label="Mean FedAvg w/ 1000 obs. per device, 2 local epochs, 5 rounds")
ax.plot(clients, final_accuracies.max(0),  "o--", color="black", lw=1, ms=5, label="Best and worst FedAvg")
ax.plot(clients, final_accuracies.min(0),  "o--", color="black", lw=1, ms=5)
# ax.axhline(71.21, ls="-", lw=2, label="Centralized learning with 5000 obs, 10 epochs")
# ax.set_ylim(45, 80)
ax.legend()
ax.grid()
fig.savefig("reports/imgs/scaling-clients.pdf")

# %%
epochs = [1, 2, 5, 10, 15, 20]
final_accuracies = np.array(
[
    [46.08, 50.78, 56.32, 56.87, 56.62, 57.82],
    [46.07, 52.39, 55.96, 55.52, 57.22, 58.26],
    [45.47, 52.28, 57.22, 58.10, 57.04, 56.89],
]
)
fig, ax = plt.subplots(figsize=figsize_std)

ax.set_title("CIFAR10: Scaling number of local epochs")
ax.set_ylabel("Accuracy on 10K test images [%] over 3 runs")
ax.set_xlabel("Number of local epochs performed on each device")
ax.plot(epochs, final_accuracies.mean(0),  "o-", color=tab_colours[2], lw=2, ms=10, label="Mean FedAvg w/ 1000 obs. per device, 5/10 dev. sampled, 5 rounds")
ax.plot(epochs, final_accuracies.max(0),  "o--", color="black", lw=1, ms=5, label="Best and worst FedAvg")
ax.plot(epochs, final_accuracies.min(0),  "o--", color="black", lw=1, ms=5)
ax.legend()
ax.grid()
fig.savefig("reports/imgs/local-epochs.pdf")

# %%
alphas = [0.001, 0.1, 10, 100]
final_accuracies = np.array(
[
    [13.61, 26.98, 52.00, 53.53],
    [11.92, 28.56, 53.13, 52.21],
    [10.01, 28.46, 52.43, 53.63],
]
)
fig, ax = plt.subplots(figsize=figsize_std)

ax.set_title("CIFAR10: Performance over data imbalance")
ax.set_ylabel("Accuracy on 10K test images [%] over 3 runs")
ax.set_xlabel("Dirichlet class balance parameter $\\alpha$")
ax.semilogx(alphas, final_accuracies.mean(0),  "o-", color=tab_colours[3], lw=2, ms=10, label="Mean FedAvg w/ 1000 obs. per device, 5/10 dev. sampled, 5 rounds")
ax.semilogx(alphas, final_accuracies.max(0),  "o--", color="black", lw=1, ms=5, label="Best and worst FedAvg")
ax.semilogx(alphas, final_accuracies.min(0),  "o--", color="black", lw=1, ms=5)
ax.legend()
ax.grid()
fig.savefig("reports/imgs/dirichlet.pdf")

# %%
gammas = [1, 0.9, 0.5, 0.1]
final_accuracies = np.array(
    [57.50, 54.11, 37.89, 25.60],
)
fig, ax = plt.subplots(figsize=figsize_std)

ax.set_title("CIFAR10: Performance over learning rate scheduling")
ax.set_ylabel("Accuracy on 10K test images [%]")
ax.set_xlabel("Learning rate scheduler multiplicative factor $\\gamma$ at learning rate 0.001")
ax.plot(gammas, final_accuracies,  "o-", color=tab_colours[4], lw=2, ms=10, label="FedAvg w/ 1000 obs. per device, 5/10 dev. sampled, 10 rounds")
ax.legend()
ax.grid()
fig.savefig("reports/imgs/lr-schedule.pdf")
