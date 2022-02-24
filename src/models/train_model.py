import os

import hydra
import torch

from src.data.make_dataset import DATA_PATH, get_mnist_dataloader
from src.models.architectures.conv import MNISTConvNet

LOG_INTERVAL = 100


def epoch(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epoch: int,
):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tMean loss: {:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    loss.item() / len(data),
                )
            )


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
):
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
    print(
        "Eval: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            loss,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    model_cfg = cfg.configs.model
    train_cfg = cfg.configs.training

    print(f"{model_cfg = }")
    print(f"{train_cfg = }")

    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Use validation set instead of test set?
    train_dataloader = get_mnist_dataloader(DATA_PATH, train_cfg.batch_size)
    test_dataloader = get_mnist_dataloader(DATA_PATH, train_cfg.batch_size, train=False)
    image_shape = train_dataloader.dataset[0][0][0].shape
    model = MNISTConvNet(input_shape=image_shape, output_size=10, **model_cfg).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(train_cfg.epochs):
        epoch(model, device, train_dataloader, optimizer, criterion, i + 1)
        evaluate(model, device, test_dataloader, criterion)

    os.makedirs(train_cfg.output_folder, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(train_cfg.output_folder, "mnist_conv.pt")
    )


if __name__ == "__main__":
    main()
