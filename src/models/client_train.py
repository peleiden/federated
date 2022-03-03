from typing import OrderedDict

import torch
from pelutils import log
from torch.functional import Tensor

from src.data.make_dataset import DATA_PATH, get_dataloader, get_mnist
from src.models.architectures.conv import MNISTConvNet
from src.models.train_model import epoch


class ClientTrainer:
    def __init__(
        self,
        train_cfg: dict,
        model_cfg: dict,
    ):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full_dataset = get_mnist(DATA_PATH, train=True)
        image_shape = self.full_dataset[0][0].shape[1:]
        output_size = len(self.full_dataset.classes)

        self.model = MNISTConvNet(
            input_shape=image_shape,
            output_size=output_size,
            **self.model_cfg,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def run_round(
        self,
        state_dict: OrderedDict[str, Tensor],
        idx: int,
        data_key: str,
        split: list[int],
    ) -> OrderedDict[str, Tensor]:
        """
        Receives model from server, trains it and returns trained version
        """
        self.model.load_state_dict(state_dict)
        log(f"Running as client {idx} with data {data_key}")
        local_dataloader = get_dataloader(
            self.full_dataset, self.train_cfg.batch_size, split
        )

        for i in range(self.train_cfg.local_epochs):
            epoch(
                self.model,
                self.device,
                local_dataloader,
                self.optimizer,
                self.criterion,
                i + 1,
            )
        return self.model.state_dict()
