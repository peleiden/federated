# %% [markdown]
# This is a Jupytext notebook, use e.g. [the vscode extension](https://marketplace.visualstudio.com/items?itemName=donjayamanne.vscode-jupytext) and open right-click + open as Jupyter Notebook on the file

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pelutils.ds.plot import figsize_std, rc_params, tab_colours
rcParams.update(rc_params)

# %%
clients = np.arange(1, 11)
final_accuracies = np.array([94.03, 94.76, 94.85, 95.30, 94.83, 95.24, 95.35, 95.43, 95.17, 94.79])
fig, ax = plt.subplots(figsize=figsize_std)

ax.set_title("Scaling over clients sampled")
ax.set_ylabel("Accuracy on 10K test images [%]")
ax.set_xlabel("Number of clients sampled each round out of 10")
ax.plot(clients, final_accuracies,  "o--", color=tab_colours[1], lw=3, ms=5, label="FedAvg with 500 obs. per device, 2 local epochs, 5 rounds")
ax.axhline(97.77, lw=3, label="Centralized learning with 5000 obs, 10 epochs")
ax.set_ylim(93, 99)
ax.legend()
ax.grid()

