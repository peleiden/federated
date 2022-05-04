from __future__ import annotations
from abc import ABC, abstractmethod
import itertools

import numpy as np
from torchvision.datasets.vision import VisionDataset

class Splitter(ABC):
    """
    Creates a list of dataset indeces for each client,
    giving each client a string id containing information about the split.
    """

    @abstractmethod
    def split(
        self, clients: int, local_data_amount: int, dataset: VisionDataset
    ) -> dict[str, list[int]]:
        raise NotImplementedError()


class EqualIIDSplit(Splitter):
    """
    Each client gets an equally-sized, randomly chosen dataset.
    """

    def split(
        self, clients: int, local_data_amount: int, dataset: VisionDataset
    ) -> dict[str, list[int]]:
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        idx = idx[: local_data_amount * clients]
        splits = np.array_split(idx, clients)
        return {f"iid-sample-{i}": split.tolist() for i, split in enumerate(splits)}

def uniformity(samples: np.ndarray):
    # Some measure of uniformity. If 0, label_sampling is perfectly uniform
    label_sampling = samples.sum(axis=0)
    return np.abs(label_sampling-label_sampling.mean()).sum() / label_sampling.size

class DirichletUnbalanced(Splitter):
    """
    Divide into non-overlapping splits where label distribution in each is Dirichlet
    """
    max_iters = 10

    def __init__(self, alpha: float):
        self.alpha = alpha

    def split(
        self, clients: int, local_data_amount: int, dataset: VisionDataset
    ) -> dict[str, list[int]]:
        L = len(dataset.classes)
        # Divide train indices into labels, limit amount and shuffle within each
        # Code assumes that dataset has same amount of each label
        idx = np.argsort([x[1] for x in dataset]).reshape(L, -1)
        [np.random.shuffle(c) for c in idx]

        # Draw dirichlet distribution
        samples = np.random.default_rng().dirichlet([self.alpha]*L, clients)
        # samples /= samples.sum()

        # Run iterative procedure, shuffling the drawn Dirichlet samples to get uniformity over
        i = 0
        unif = uniformity(samples)
        prev_unif = unif + 1  # Add one to get loop going
        while abs(unif - prev_unif) > 1e-8 and i <= self.max_iters:
            prev_unif = unif
            diffs = samples.sum(axis=0) - clients / L
            over_sampled_labels, = np.where(diffs > 0)
            under_sampled_labels, = np.where(diffs <= 0)
            best_uniformity = uniformity(samples)
            best_uniformity_idcs = None
            for j in range(clients):
                for idx1, idx2 in itertools.product(over_sampled_labels, under_sampled_labels):
                    s_copy = samples.copy()
                    tmp = s_copy[j, idx1]
                    s_copy[j, idx1] = s_copy[j, idx2]
                    s_copy[j, idx2] = tmp
                    unif = uniformity(s_copy)
                    if unif < best_uniformity:
                        best_uniformity = unif
                        best_uniformity_idcs = (j, idx1, idx2)
            if best_uniformity_idcs is not None:
                j, idx1, idx2 = best_uniformity_idcs
                tmp = samples[j, idx1]
                samples[j, idx1] = samples[j, idx2]
                samples[j, idx2] = tmp
            i += 1
            unif = uniformity(samples)
        samples = samples * (clients / idx.shape[0]) / samples.sum(0).max()
        # Convert the (somewhat) normalized samples to class amounts
        label_amounts = (samples * local_data_amount).astype(int)
        # Divide amounts first amongst labels, then joining up on each client
        label_splits = [np.split(idx[i], division) for i, division in enumerate(label_amounts.cumsum(0).T)]
        splits = [np.concatenate([l[i] for l in label_splits]) for i in range(clients)]
        # Shuffle client arrays
        [np.random.shuffle(s) for s in splits]
        return {f"dirichlet-sample-{i}": split.tolist() for i, split in enumerate(splits)}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.data.make_dataset import get_cifar10, DATA_PATH

    SMALL_SIZE = 40
    MEDIUM_SIZE = 40
    BIGGER_SIZE = 44

    plt.figure(figsize=(11, 8))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Generate example data
    dataset = get_cifar10(DATA_PATH, train=True)
    alpha, clients, local_data_amount = 100, 40, 1250
    split = DirichletUnbalanced(alpha).split(clients, local_data_amount, dataset)
    # split = EqualIIDSplit().split(clients, local_data_amount, dataset)
    lens = [len(x) for x in split.values()]
    print(f"Total amount {sum(lens)}, max amount {max(lens)}, min amount {min(lens)}.")

    # Visualize dirichilet
    K = len(dataset.classes)
    labels = [[dataset[i][1] for i in idx] for idx in split.values()]
    pos = np.zeros((clients, K))

    for i in range(clients):
        for label, count in zip(*np.unique(labels[i], return_counts=True)):
            pos[i, label] = count
    pos /= pos.sum(1, keepdims=True)
    pos *= 100
    left = np.zeros(clients)
    for i in range(K):
        plt.barh(range(clients), pos[:, i], left=left)
        left += pos[:, i]


    # for i in range
    plt.ylabel("Client")
    plt.xlabel("Label distribution [%]")
    plt.xlim(-2, 102)
    plt.title(f"Dirichlet ($\\alpha={alpha}$)")
    plt.tight_layout()
    plt.savefig(f"reports/imgs/splits({alpha=}).pdf")
    # plt.title(f"IID")
    # plt.savefig(f"reports/imgs/splits(IID).pdf")
