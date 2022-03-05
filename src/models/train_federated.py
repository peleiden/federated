from __future__ import annotations
from threading import Thread
from typing import Generator, Optional, OrderedDict
import json
import os

from pelutils import log
import hydra
import requests

from src.client_utils import state_dict_from_base64, state_dict_to_base64
from src.models.client_train import ClientTrainer
from src.models.server_train import ServerTrainer
from src.models.train_model import evaluate


def setup_local_clients(start_args: dict, num_clients: int) -> tuple[ClientTrainer]:
    return tuple(ClientTrainer(**start_args) for _ in range(num_clients))

def run_local_rounds(clients: tuple[ClientTrainer], client_args: list) -> Generator[OrderedDict, None, None]:
    for i, args in enumerate(client_args):
        yield clients[i].run_round(**args)

def setup_external_clients(ip: str, start_args: dict, num_devices: int):
    def setup_single_client(num: int):
        log("Sending training configuration to device %i" % num)
        response = requests.post(f"http://{ip}:{3080+num}/configure-training", json=dict(
            train_cfg=dict(start_args["train_cfg"]),
            model_cfg=dict(start_args["model_cfg"]),
        ))
        log("Got status code %i" % response.status_code)
        log(response.content)
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i" % (num, response.status_code))

    threads = list()
    for i in range(num_devices):
        threads.append(Thread(target=lambda: setup_single_client(i)))
        threads[-1].start()
    for i in range(num_devices):
        threads[i].join()

def run_external_rounds(ip: str, client_args: list) -> Generator[OrderedDict, None, None]:
    returned_b64s: list[str] = [None] * len(client_args)
    def train_single_client(num: int, args: dict):
        args = args.copy()
        args["state_dict"] = state_dict_to_base64(args["state_dict"])
        log("Sending state dict to device %i" % num)
        response = requests.post(f"http://{ip}:{3080+num}/train-round", json=args)
        log("Got status code %i from device %i" % (response.status_code, num))
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i" % (num, response.status_code))
        returned_b64s[num] = json.loads(response.content)["data"]

    threads = list()
    for i, args in enumerate(client_args):
        threads.append(Thread(target=lambda: train_single_client(i, args)))
        threads[-1].start()
    for i in range(len(client_args)):
        threads[i].join()
        yield state_dict_from_base64(returned_b64s[i])

@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    server = ServerTrainer(cfg)
    start_args = server.get_client_start_args()

    ip: Optional[str] = os.environ.get("IP")
    if ip:
        log("Using clients at IP %s" % ip)
        setup_external_clients(ip, start_args, server.train_cfg.clients_per_round)
    else:
        log("Using local training")
        clients = setup_local_clients(start_args, server.train_cfg.clients)

    for i in range(cfg.configs.training.communication_rounds):
        client_args = server.get_communication_round_args()
        log(f"Round {i}. Chose clients with idx", list(c["idx"] for c in client_args))

        if ip:
            received_data = list(run_external_rounds(ip, client_args))
        else:
            received_data = list(run_local_rounds(clients, client_args))

        server.aggregate(received_data)
        evaluate(server.model, server.device, server.test_dataloader, server.criterion)

if __name__ == "__main__":
    log.configure("training.log")  # Hydra controls cwd
    main()
