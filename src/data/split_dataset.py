from abc import ABC, abstractmethod
import random

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

class DirichletUnbalanced(Splitter):
    """
    Divide into non-overlapping splits where label distribution in each is Dirichlet
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def split(
        self, clients: int, local_data_amount: int, dataset: VisionDataset
    ) -> dict[str, list[int]]:
        K = len(dataset.classes)
        # Divide train indices into labels, limit amount and shuffle within each
        # Code assumes that dataset has same amount of each label
        idx = np.argsort([x[1] for x in dataset]).reshape(K, -1)
        # Draw dirichlet distribution and and convert to split indeces
        K = 4
        label_dists = np.random.default_rng().dirichlet([self.alpha]*K, clients)

        classlim = (local_data_amount*clients)
        split_pos = (label_dists.cumsum(1) * classlim).astype(int)
        # Split each label between clients
        splits = [[] for _ in range(clients)]
        for i in range(K):
            for j, split in enumerate(np.split(idx[i], split_pos[i])[:-1]):
                splits[j].extend(split)
        # Shuffle and limit data for each agent
        [random.shuffle(s) for s in splits]
        return {f"dirichlet-sample-{i}": split[:local_data_amount] for i, split in enumerate(splits)}

# Non-overlappende data
# Man skal kunne 10 5000

if __name__ == "__main__":
    # Visualize dirichilet
    import matplotlib.pyplot as plt
    from src.data.make_dataset import get_cifar10, DATA_PATH

    dataset = get_cifar10(DATA_PATH, train=False)
    alpha, clients, local_data_amount = 0.1, 3, 1000
    split = DirichletUnbalanced(alpha).split(clients, local_data_amount, dataset)
    K = len(dataset.classes)
    labels = [[dataset[i][1] for i in idx] for idx in split.values()]
    pos = np.zeros((clients, K))

    for i in range(clients):
        for label, count in zip(*np.unique(labels[i], return_counts=True)):
            pos[i, label] = count
    left = np.zeros(clients)
    for i in range(K):
        plt.barh(range(clients), pos[:, i], left=left)
        left += pos[:, i]

    # for i in range
    plt.ylabel("Client")
    plt.xlabel("Label")
    plt.title(f"Dirichlet {alpha=}")
    plt.savefig("splits.png")
