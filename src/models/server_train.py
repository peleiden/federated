from typing import Any, OrderedDict

import numpy as np
import torch
from omegaconf import open_dict
from pelutils import log
from torch.functional import Tensor

from src.data.make_dataset import DATA_PATH, get_dataloader, get_mnist
from src.data.split_dataset import EqualIIDSplit
from src.models.architectures.conv import MNISTConvNet


class ServerTrainer:
    def __init__(self, cfg: dict):
        self.model_cfg = cfg.configs.model
        self.train_cfg = cfg.configs.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = get_dataloader(
            get_mnist(DATA_PATH, train=True), self.train_cfg.batch_size
        )
        self.test_dataloader = get_dataloader(
            get_mnist(DATA_PATH, train=False), self.train_cfg.batch_size
        )
        image_shape = self.train_dataloader.dataset[0][0][0].shape
        output_size = len(self.test_dataloader.dataset.classes)
        self.model = MNISTConvNet(
            input_shape=image_shape, output_size=output_size, **self.model_cfg
        ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        splitter = EqualIIDSplit()
        self.splits = splitter.split(
            self.train_cfg.clients,
            self.train_cfg.local_data_amount,
            self.train_dataloader.dataset,
        )
        self.idx_to_split = list(self.splits.keys())

        log("Created server trainer.")
        log(f"{self.model_cfg = }")
        log(f"{self.train_cfg = }")

    def get_client_start_args(self) -> dict[str, Any]:
        """
        Generate a list of start args for each clients where args correspond to those given to build_from_start
        """
        return dict(
            train_cfg=self.train_cfg,
            model_cfg=self.model_cfg,
        )

    def get_communication_round_args(self) -> list[dict[str, Any]]:
        """
        Chooses a numer of clients and prepares args for them.
        args should fit with ClientTrainer.train()
        """
        client_args = list()
        for i in np.random.choice(
            self.train_cfg.clients, self.train_cfg.clients_per_round, replace=False
        ):
            data_key = self.idx_to_split[i]
            client_args.append(
                dict(
                    state_dict=self.model.state_dict(),
                    idx=int(i),
                    data_key=data_key,
                    split=self.splits[data_key],
                )
            )
        return client_args

    def aggregate(self, received_models: list[OrderedDict[str, Tensor]]):
        out_dict = self.model.state_dict()
        # FedAVG
        for key in out_dict:
            out_dict[key] = torch.stack([m[key] for m in received_models]).mean(0)
        self.model.load_state_dict(out_dict)
