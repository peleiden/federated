from __future__ import annotations
from typing import OrderedDict

import torch
from pelutils import log
from torch.functional import Tensor

from src.data.make_dataset import DATA_PATH, get_dataloader, get_mnist, get_cifar10
from src.models.architectures.conv import SimpleConv
from src.models.train_model import epoch, evaluate


class ClientTrainer:
    def __init__(
        self,
        train_cfg: dict,
        model_cfg: dict,
        data_path=None,
    ):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        get_data = get_cifar10 if self.train_cfg["dataset"] == "cifar10" else get_mnist
        self.train_dataset = get_data(data_path or DATA_PATH, train=True)
        if self.train_cfg["local_eval"]:
            self.test_dataset = get_data(data_path or DATA_PATH, train=False)
        image_shape = self.train_dataset[0][0].shape
        output_size = len(self.train_dataset.classes)

        self.model = SimpleConv(
            input_shape=image_shape,
            output_size=output_size,
            **self.model_cfg,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"])
        self.original_state_dict = self.optimizer.state_dict().copy()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.train_cfg["lr_gamma"])

    def run_round(
        self,
        state_dict: OrderedDict[str, Tensor],
        idx: int,
        data_key: str,
        split: list[int],
        noisy_images: float,
    ) -> tuple[OrderedDict[str, Tensor], list[float], list[float]]:
        """
        Receives model from server, trains it and returns trained version
        """
        self.model.load_state_dict(state_dict)
        log(f"Running as client {idx} with data {data_key}")
        local_dataloader = get_dataloader(
            self.train_dataset, self.train_cfg["batch_size"], split
        )
        if self.train_cfg["local_eval"]:
            local_test_dataloader = get_dataloader(self.test_dataset, self.train_cfg["batch_size"])

        accs, losses, train_accs = list(), list(), list()

        for i in range(self.train_cfg["local_epochs"]):
            train_acc = epoch(
                self.model,
                self.device,
                local_dataloader,
                self.optimizer,
                self.criterion,
                i + 1,
                noisy_images=noisy_images,
            )
            train_accs.append(train_acc)
            self.scheduler.step()
            if self.train_cfg["local_eval"]:
                log.debug("Evaluating model")
                acc, loss = evaluate(self.model, self.device, local_test_dataloader, self.criterion)
                accs.append(acc)
                losses.append(loss.item())
                log.debug(
                    "Accuracy: %.4f" % acc,
                    "Loss:     %.4f" % loss.item(),
                )
            self.model.zero_grad()
            self.optimizer.load_state_dict(self.original_state_dict)
        return self.model.state_dict(), accs, losses, train_accs
