from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np


from texttable import Texttable
from latextable import draw_latex
from src.models.train_federated import Results

NAMES = dict(alpha=r"Class balance ($\alpha$)", clients_per_round="Clients sampled", noisy_clients="Noisy clients", local_epochs="Local epochs")

def results_and_cfgs(path: str):
    jobs = os.listdir(path)
    res = [Results.load(os.path.join(path, j)) for j in jobs if j.isdigit()]
    return res, [r.cfg.configs.training for r in res]

def table_over_arg():
    for var in os.listdir():
        if var == "iid":
            continue
        results, configs = results_and_cfgs(var)
        values = np.array([cfg[var] for cfg in configs])
        if var == "alpha":
            r2, c2 =  results_and_cfgs("iid")
            results.extend(r2)
            configs.extend(c2)
            values = np.concatenate((values, ["iid"]*len(r2)))
        perfs = np.array([r.test_accuracies[-1] for r in results])
        uniques = np.unique(values)
        means = np.array([np.mean(perfs[values == v]) for v in uniques])
        stds = np.array([np.std(perfs[values == v], ddof=1) for v in uniques])

        t = Texttable()
        t.set_deco(t.HEADER)
        header = (NAMES[var], *[str(x) for x in uniques])
        t.header(header)
        t.add_row(["Test acc. [\%]"] + [f"${m:.1f}"r" \pm " + f"{s:.1f}$" for m, s in zip(means, stds)])
        t.draw()
        out = draw_latex(t, caption="", label=f"tab:{var}")
        print(out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("--arg")
    args = parser.parse_args()
    os.chdir(args.location)
    table_over_arg()
