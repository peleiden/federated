from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

DATA_PATH = os.path.join(sys.path[0], "..", "..", "data")
EACH_LABEL = 2000


def get_dataloader(
    dataset: VisionDataset,
    batch_size: int,
    split: Optional[list[int]] = None,
) -> DataLoader:
    if split is not None:
        dataset = Subset(dataset, split)
    # Shuffle if we're on training data
    train = dataset.dataset.train if isinstance(dataset, Subset) else dataset.train

    return DataLoader(dataset, batch_size, shuffle=train)


def get_mnist(
    data_folder: str, train: bool = True, download: bool = False
) -> VisionDataset:
    # Params taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),
        ]
    )
    dataset = datasets.MNIST(
        data_folder, train=train, download=download, transform=transform
    )
    targets = np.array(dataset.targets)
    indices = list()
    for i in range(len(dataset.classes)):
        indices.extend(np.where(targets==i)[0].tolist()[:EACH_LABEL])
    dataset = Subset(dataset, indices).dataset
    return dataset


def get_cifar10(
    data_folder: str, train: bool = True, download: bool = False, cifar100: bool = False
) -> VisionDataset:
    # Params taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset_cls = datasets.CIFAR100 if cifar100 else datasets.CIFAR10
    dataset = dataset_cls(
        data_folder, train=train, download=download, transform=transform
    )
    return dataset


if __name__ == "__main__":
    # Call this once to download to disk
    print("Downloading data to disk if not already there ...")
    # MNIST
    get_mnist(DATA_PATH, train=True, download=True)
    get_mnist(DATA_PATH, train=False, download=True)
    # CIFAR-10
    get_cifar10(DATA_PATH, train=True, download=True)
    get_cifar10(DATA_PATH, train=False, download=True)
    # CIFAR-100
    get_cifar10(DATA_PATH, train=True, download=True, cifar100=True)
    get_cifar10(DATA_PATH, train=False, download=True, cifar100=True)
    print("Done.")
