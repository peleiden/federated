import os

import hydra
from pelutils import log

from src.models.client_train import ClientTrainer
from src.models.server_train import ServerTrainer
from src.models.train_model import evaluate


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    server = ServerTrainer(cfg)
    start_args = server.get_client_start_args()

    clients = [ClientTrainer(**start_args) for _ in range(server.train_cfg.clients)]
    for i in range(cfg.configs.training.communication_rounds):
        client_args = server.get_communication_round_args()
        log(f"Round {i}. Chose clients with idx", list(c["idx"] for c in client_args))

        received_data = list()
        for i, args in enumerate(client_args):
            received_data.append(clients[i].run_round(**args))
        server.aggregate(received_data)
        evaluate(server.model, server.device, server.test_dataloader, server.criterion)


if __name__ == "__main__":
    log.configure("training.log")  # Hydra controls cwd
    main()
