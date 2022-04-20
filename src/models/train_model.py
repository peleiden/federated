from __future__ import annotations
import os

import hydra
import torch
import wandb
from pelutils import log

from src.data.make_dataset import DATA_PATH, get_dataloader, get_mnist, get_cifar10
from src.models.architectures.conv import SimpleConv

LOG_INTERVAL = 1


def epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epoch: int,
    use_wandb: bool = False,
    noisy_images: float = 0,
) -> float:
    """ Returns mean training accuracy """
    model.train()
    train_acc = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        num_noisy_images = int(noisy_images*len(target))
        target = target.clone()
        target[torch.randperm(len(target))[:num_noisy_images]] = torch.randint(0, 10, (num_noisy_images,), device=target.device)
        log.debug("Scrambled %i labels" % num_noisy_images)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_acc += (torch.softmax(output, dim=-1).argmax(dim=-1) == target)\
            .to(float)\
            .mean()\
            .item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            update = batch_idx * len(data)
            mean_loss = loss.item() / len(data)
            log.debug(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tMean loss: {:.4f}".format(
                    epoch,
                    update,
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    mean_loss,
                )
            )
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "update": update * epoch + update,
                        "loss": mean_loss,
                    }
                )
        del output, loss
    return train_acc / (batch_idx+1)

def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    use_wandb: bool = False,
) -> tuple[float, float]:
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            pred = output.argmax(1, keepdim=True)
            correct += (pred == target.view_as(pred)).sum().item()
    loss /= len(dataloader.dataset)
    acc = 100 * correct / len(dataloader.dataset)
    log(
        "Eval: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            loss,
            correct,
            len(dataloader.dataset),
            acc,
        )
    )
    if use_wandb:
        wandb.log({"eval_loss": loss, "eval_acc": acc})

    return acc, loss


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    model_cfg = cfg.configs.model
    train_cfg = cfg.configs.training

    log(f"{model_cfg = }")
    log(f"{train_cfg = }")

    wandb.init(config=cfg, project="Federated Learning")

    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_data = get_cifar10 if train_cfg["dataset"] == "cifar10" else get_mnist

    train_dataloader = get_dataloader(
        get_data(DATA_PATH, train=True), train_cfg.batch_size
    )
    test_dataloader = get_dataloader(
        get_data(DATA_PATH, train=False), train_cfg.batch_size
    )

    image_shape = train_dataloader.dataset[0][0].shape
    model = SimpleConv(input_shape=image_shape, output_size=10, **model_cfg).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(train_cfg.local_epochs):
        epoch(
            model, device, train_dataloader,
            optimizer, criterion, i + 1, use_wandb=True,
        )
        evaluate(model, device, test_dataloader, criterion, use_wandb=True)

    torch.save(model.state_dict(), "mnist_conv.pt")


if __name__ == "__main__":
    log.configure("training.log")  # Hydra controls cwd
    main()
