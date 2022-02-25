from abc import ABC, abstractmethod

import numpy as np
from torchvision.datasets.vision import VisionDataset


class Splitter(ABC):
    """
    Creates a list of dataset indeces for each client,
    giving each client a string id containing information about the split.
    """

    @abstractmethod
    def split(self, clients: int, dataset: VisionDataset) -> dict[str, list[int]]:
        raise NotImplementedError()


class EqualIIDSplit(Splitter):
    """
    Each client gets an equally-sized, randomly chosen dataset.
    """

    def split(self, clients: int, dataset: VisionDataset) -> dict[str, list[int]]:
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        splits = np.array_split(idx, clients)
        return {f"iid-sample-{i}": list(split) for i, split in enumerate(splits)}
