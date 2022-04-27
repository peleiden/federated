from __future__ import annotations
from typing import Any, OrderedDict
from copy import deepcopy

import numpy as np
import torch
from pelutils import log
import torch.nn.functional as F
from torch.functional import Tensor

from src.data.make_dataset import DATA_PATH, get_dataloader, get_mnist, get_cifar10
from src.data.split_dataset import DirichletUnbalanced, EqualIIDSplit
from src.models.architectures.conv import SimpleConv


class ServerTrainer:
    def __init__(self, cfg: dict):
        self.model_cfg = cfg.configs.model
        self.train_cfg = cfg.configs.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        get_data = get_cifar10 if self.train_cfg["dataset"] == "cifar10" else get_mnist
        self.train_dataloader = get_dataloader(
            get_data(DATA_PATH, train=True), self.train_cfg["batch_size"],
        )
        self.test_dataloader = get_dataloader(
            get_data(DATA_PATH, train=False), self.train_cfg["batch_size"],
        )

        image_shape = self.train_dataloader.dataset[0][0].shape
        output_size = len(self.test_dataloader.dataset.classes)
        self.model = SimpleConv(
            input_shape=image_shape, output_size=output_size, **self.model_cfg
        ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        splitter = DirichletUnbalanced(self.train_cfg["alpha"]) if self.train_cfg["split"] == "dirichlet" else EqualIIDSplit()
        self.splits = splitter.split(
            self.train_cfg.clients,
            self.train_cfg.local_data_amount,
            self.train_dataloader.dataset,
        )
        self.idx_to_split = list(self.splits.keys())

        if self.train_cfg["aggregation"] == "feddf":
            assert self.train_cfg["dataset"] == "cifar10", "Currently, distil dataset is only supplied for CIFAR-10"
            self.distil_dataloader = get_dataloader(
                get_cifar10(DATA_PATH, train=True, cifar100=True), self.train_cfg["distil"]["batch_size"],
            )
            self.distil_criterion = torch.nn.KLDivLoss(reduction="batchmean")
            self.distil_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["distil"]["lr"])

        log("Applying %.4f %% noise to %i clients" % (100 * self.train_cfg.noisy_images, self.train_cfg.noisy_clients))
        self.noisy_clients = set(np.random.choice(np.arange(self.train_cfg.clients), size=self.train_cfg.noisy_clients, replace=False))
        log("The following clients are noisy:", self.noisy_clients, with_info=False)

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
                    noisy_images=(i in self.noisy_clients)*self.train_cfg.noisy_images,
                )
            )
        return client_args

    def aggregate(self, received_models: list[OrderedDict[str, Tensor]]):
        if self.train_cfg["aggregation"] == "feddf":
            self.fed_df(received_models)
        else:
            self.fed_avg(received_models)

    def fed_avg(self, received_models: list[OrderedDict[str, Tensor]]):
        # Simple average over all model tensors
        out_dict = self.model.state_dict()
        for key in out_dict:
            out_dict[key] = torch.stack([m[key] for m in received_models]).mean(0)
        self.model.load_state_dict(out_dict)

    def fed_df(self, received_models: list[OrderedDict[str, Tensor]]):
        # Run ensemble distillation
        log(f"Running ensemble distillation for {self.train_cfg['distil']['steps']} steps")
        teachers = list()
        for weights in received_models:
            teacher = deepcopy(self.model) # <- Assuming client and server have same arch
            teacher.load_state_dict(weights)
            teacher.eval()
            teachers.append(teacher)
        # Initialize learner as average
        self.fed_avg(received_models)

        # Run distillation
        for i, (data, _) in enumerate(self.distil_dataloader):
            self.distil_optimizer.zero_grad()
            if i == self.train_cfg["distil"]["steps"]:
                break
            data = data.to(self.device)
            with torch.no_grad():
                teacher_logits = torch.stack([teacher(data) for teacher in teachers])
            target_probs = F.softmax(teacher_logits.mean(0), dim=-1)
            preds = F.log_softmax(self.model(data), dim=-1)

            loss = self.distil_criterion(preds, target_probs)
            loss.backward()
            self.distil_optimizer.step()

            log.debug("Distillation Loss {}: {:.4f}".format(i, loss.item()/len(data)))
