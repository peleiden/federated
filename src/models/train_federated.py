import os

import hydra
from pelutils import log

from src.models.client_train import ClientTrainer
from src.models.server_train import ServerTrainer
from src.models.train_model import evaluate


@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    server = ServerTrainer(cfg)

    clients = [
        ClientTrainer.build_from_start(**args)
        for args in server.get_client_start_args()
    ]
    for i in range(cfg.configs.training.communication_rounds):
        client_args = server.get_communication_round_args()
        log(f"Round {i}. Chose clients with idx {sorted(list(client_args.keys()))}")

        received_data = list()
        for j, args in client_args.items():
            log("Running local round for client", j)
            received_data.append(clients[j].train(**args))
        server.aggregate(received_data)
        evaluate(server.model, server.device, server.test_dataloader, server.criterion)


if __name__ == "__main__":
    log.configure("training.log") # Hydra controls cwd
    main()
