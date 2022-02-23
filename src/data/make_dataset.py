import os
import sys

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_PATH = os.path.join(sys.path[0], "..", "..", "data")


def get_mnist_dataloader(
    data_folder: str, batch_size: int, train: bool = True, download=False
) -> DataLoader:
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
    return DataLoader(dataset, batch_size, shuffle=train)


if __name__ == "__main__":
    # Call this once to download to disk
    print("Downloading data to disk if not already there ...")
    get_mnist_dataloader(DATA_PATH, 1, train=True, download=True)
    get_mnist_dataloader(DATA_PATH, 1, train=False, download=True)
    print("Done.")
