from typing import OrderedDict

import torch
from pelutils import log
from torch.functional import Tensor

from src.data.make_dataset import DATA_PATH, get_mnist_dataloader
from src.models.architectures.conv import MNISTConvNet
from src.models.train_model import epoch


class ClientTrainer:
    def __init__(
        self,
        train_cfg: dict,
        model_cfg: dict,
        dataloader: torch.utils.data.DataLoader,
    ):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_shape = self.dataloader.dataset[0][0][0].shape
        output_size = len(self.dataloader.dataset.dataset.classes)

        self.model = MNISTConvNet(
            input_shape=image_shape,
            output_size=output_size,
            **self.model_cfg,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @classmethod
    def build_from_start(
        cls,
        train_cfg: dict,
        model_cfg: dict,
        split: list[int],
    ):
        local_dataloader = get_mnist_dataloader(
            DATA_PATH, train_cfg.batch_size, split=split
        )
        log(f"Creating local trainer with dataset of size {len(split)}")
        log(f"{train_cfg = }")
        return cls(train_cfg, model_cfg, local_dataloader)

    def train(self, state_dict: OrderedDict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """
        Receives model from server, trains it and returns trained version
        """
        self.model.load_state_dict(state_dict)
        for i in range(self.train_cfg.local_epochs):
            epoch(
                self.model,
                self.device,
                self.dataloader,
                self.optimizer,
                self.criterion,
                i + 1,
            )
        return self.model.state_dict()
