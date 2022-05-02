from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np


from texttable import Texttable
from latextable import draw_latex
from src.models.train_federated import Results

NAMES = dict(
    local_epochs="Local epochs ($E$)",
    clients_per_round="Clients sampled ($S$)",
    alpha=r"Class balance ($\alpha$)",
    noisy_clients="Noisy clients ($N_K$)",
)


def results_and_cfgs(path: str):
    jobs = os.listdir(path)
    res = [Results.load(os.path.join(path, j)) for j in jobs if j.isdigit()]
    return res, [r.cfg.configs.training for r in res]


def table_over_arg():
    M, S, V, U = list(), list(), list(), list()
    for var in NAMES.keys():
        if var == "iid":
            continue
        results, configs = results_and_cfgs(var)
        values = np.array([cfg[var] for cfg in configs])
        if var == "alpha":
            r2, c2 = results_and_cfgs("iid")
            results.extend(r2)
            configs.extend(c2)
            values = np.concatenate((values, ["iid"] * len(r2)))
        perfs = np.array([r.test_accuracies[-1] for r in results])
        uniques = np.unique(values)
        M.append(np.array([np.mean(perfs[values == v]) for v in uniques]))
        S.append(np.array([np.std(perfs[values == v], ddof=1) for v in uniques]))
        V.append(var)
        U.append(uniques)
    start, end = (
        r"""
\begin{table}[htb!]
    \centering
    \begin{tabular}""",
        r"""\end{tabular}
    \caption{
    Final test accuracies [\%] of FedAvg models over $K=40$ clients when running for $L=20$ communication rounds.
    Each run is repeated five times to produce an approximate 95\%\ confidence interval.
    }
    \label{tab:main}
\end{table}""",
    )
    r = max(x.size for x in M)
    mc = lambda x: r"        \multicolumn{" + f"{r}" + r"}{c}{" + f"{x}" + r"}\\" + "\n"
    s = start + "{" f"{'l'*r}" "}\n"
    s += r"       \hline" "\n"
    for means, stds, var, uniques in zip(M, S, V, U):
        s += mc(NAMES[var])
        s += "        " + " & ".join(f"{x}" for x in uniques) + r" \\" + "\n"
        s += r"       \hline" "\n"
        s += (
            "        "
            + " & ".join(
                f"${m:.1f}" r" \pm " + f"{1.96 / len(stds)**0.5 * s:.1f}$"
                for m, s in zip(means, stds)
            )
            + r" \\"
            + "\n"
        )
    s += end
    print(s)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("--arg")
    args = parser.parse_args()
    os.chdir(args.location)
    table_over_arg()
