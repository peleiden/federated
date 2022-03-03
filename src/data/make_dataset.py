import os
import sys
from typing import Optional

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

DATA_PATH = os.path.join(sys.path[0], "..", "..", "data")


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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
        ]
    )
    dataset = datasets.MNIST(
        data_folder, train=train, download=download, transform=transform
    )
    return dataset


if __name__ == "__main__":
    # Call this once to download to disk
    print("Downloading data to disk if not already there ...")
    get_mnist(DATA_PATH, train=True, download=True)
    get_mnist(DATA_PATH, train=False, download=True)
    print("Done.")
