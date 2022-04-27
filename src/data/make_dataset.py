from __future__ import annotations
from ast import Tuple

import os
import pickle
import sys
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

DATA_PATH = os.path.join(sys.path[0], "..", "..", "data")


def rgb2grey(im):
    im = np.round(0.299 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2])
    im[im<0] = 0
    im[im>255] = 255
    return im.astype(np.uint8)

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                entry["data"] = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                entry["data"] = rgb2grey(entry["data"])

                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data)

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


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
            transforms.Normalize((0.4808609379595587,), (0.23919066003980183,)),
        ]
    )
    dataset_cls = CIFAR100 if cifar100 else CIFAR10
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
