from __future__ import annotations
from dataclasses import dataclass
from threading import Thread
from typing import Generator, Optional, OrderedDict
import json
import os
import time

from pelutils import log, TickTock, DataStorage, Levels
import hydra
import requests

from src.client_utils import state_dict_from_base64, state_dict_to_base64
from src.models.client_train import ClientTrainer
from src.models.server_train import ServerTrainer
from src.models.train_model import evaluate


_ping_telemetry = True

@dataclass
class Results(DataStorage):
    # Training configuration
    cfg: dict
    start_args: dict
    # List of timings for each round
    # [
        # {
            # total_train: float,  # Total training time for this round
            # total_test: float,  # Part of round spent evaluating
            # The following are per device. They are empty if ip is None
            # response_time: list[float],  # Time from request is sent until return
            # decode: list[float],  # Time spent decoding state dict
            # train:  list[float],  # Time spent training
            # encode: list[float],  # Time spent encoding state dict
        # }
    # ]
    timings: list[dict[str, float | list[float]]]
    # Telemetry for each device
    # [
        # {
            # "timestamp": list[float],  # When the telemetry responses were received
            # "memory_usage": list[float],  # Percentage of memory used on device
        # }
    # ]
    telemetry: list[dict[str, list[float]]]
    # Array of accuracies after each round
    eval_timestamps: list[float]  # Measured immediately after evaluation of aggregated model
    test_accuracies: list[float]
    test_losses: list[float]

    ip: str | None

    json_name = "results.json"

def setup_local_clients(start_args: dict, num_clients: int) -> tuple[ClientTrainer]:
    return tuple(ClientTrainer(**start_args) for _ in range(num_clients))

def run_local_rounds(clients: tuple[ClientTrainer], client_args: list) -> Generator[OrderedDict, None, None]:
    for i, args in enumerate(client_args):
        yield clients[i].run_round(**args)

def setup_external_clients(ip: str, start_args: dict, num_devices: int, training_id: int) -> list[float]:
    telemetries = [None] * num_devices
    def setup_single_client(num: int):
        log("Requesting initial telemetry from device %i" % num)
        response = requests.get(f"http://{ip}:{3080+num}/telemetry")
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i" % (num, response.status_code))
        telemetries[num] = json.loads(response.content)["data"]

        log("Sending training configuration to device %i" % num)
        response = requests.post(f"http://{ip}:{3080+num}/configure-training", json=dict(
            train_cfg=dict(start_args["train_cfg"]),
            model_cfg=dict(start_args["model_cfg"]),
            training_id = training_id,
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

    return telemetries

def run_external_rounds(ip: str, client_args: list, training_id: int) -> Generator[tuple[OrderedDict, dict[str, float]], None, None]:
    returned_b64s: list[str] = [None] * len(client_args)
    returned_timings: list[dict[str, float]] = [None] * len(client_args)

    def train_single_client(num: int, args: dict):
        args = args.copy()
        args["state_dict"] = state_dict_to_base64(args["state_dict"])
        args["training_id"] = training_id
        log("Sending state dict to device %i" % num)
        tt = TickTock()
        tt.tick()
        response = requests.post(f"http://{ip}:{3080+num}/train-round", json=args)
        response_time = tt.tock()
        log("Got status code %i from device %i" % (response.status_code, num))
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i" % (num, response.status_code))
        response = json.loads(response.content)
        returned_b64s[num] = response["data"]["state_dict"]
        returned_timings[num] = response["data"]["timings"]
        returned_timings[num]["response_time"] = response_time

    threads = list()
    for i, args in enumerate(client_args):
        threads.append(Thread(target=lambda: train_single_client(i, args)))
        threads[-1].start()
    for i in range(len(client_args)):
        threads[i].join()
        yield state_dict_from_base64(returned_b64s[i]), returned_timings[i]

def ping_telemetry(ip: str, num_clients: int, results: Results):
    global _ping_telemetry
    current_client = 0
    while _ping_telemetry:
        log.debug("Requesting telemetry from client %i" % current_client)
        response = requests.get(f"http://{ip}:{3080+current_client}/telemetry")
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i" % (current_client, response.status_code))
        results.telemetry[current_client]["timestamp"].append(time.time())
        results.telemetry[current_client]["memory_usage"].append(json.loads(response.content)["data"]["total-memory-usage-pct"])
        log.debug("Device %i reported %.2f %% memory usage" % (current_client, results.telemetry[current_client]["memory_usage"][-1]))
        current_client = (current_client + 1) % num_clients

@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    global _ping_telemetry
    server = ServerTrainer(cfg)
    start_args = server.get_client_start_args()

    tt = TickTock()

    ip: Optional[str] = os.environ.get("IP")

    results = Results(
        cfg             = cfg,
        start_args      = start_args,
        timings         = list(),
        telemetry       = list(),
        eval_timestamps = list(),
        test_accuracies = list(),
        test_losses     = list(),
        ip              = ip,
    )

    if ip:
        log("Using clients at IP %s" % ip)
        timestamp = time.time()
        training_id = hash(timestamp)
        log("Training ID: %i" % training_id)
        telemetry_readings = setup_external_clients(ip, start_args, server.train_cfg.clients_per_round, training_id)
        for device_id in range(server.train_cfg.clients_per_round):
            results.telemetry.append({
                "timestamp": [timestamp],
                "memory_usage": [telemetry_readings[device_id]["total-memory-usage-pct"]]
            })
        telemetry_thread = Thread(target=lambda: ping_telemetry(ip, server.train_cfg.clients_per_round, results))
        telemetry_thread.start()
    else:
        log("Using local training")
        clients = setup_local_clients(start_args, server.train_cfg.clients)

    for i in range(cfg.configs.training.communication_rounds):
        tt.tick()
        client_args = server.get_communication_round_args()
        log.section(f"Round {i}. Chose clients with idx", list(c["idx"] for c in client_args))

        if ip:
            received_data, timings = zip(*run_external_rounds(ip, client_args, training_id))
        else:
            received_data = list(run_local_rounds(clients, client_args))
        server.aggregate(received_data)
        train_time = tt.tock()

        tt.tick()
        acc, loss = evaluate(server.model, server.device, server.test_dataloader, server.criterion)
        test_time = tt.tock()

        results.eval_timestamps.append(time.time())
        results.test_accuracies.append(float(acc))
        results.test_losses.append(float(loss))
        results.timings.append(dict(
            total_train = train_time,
            total_test  = test_time,
        ))
        for device_timing_key in "response_time", "decode", "train", "encode":
            if ip:
                results.timings[-1][device_timing_key] = [t[device_timing_key] for t in timings]
            else:
                results.timings[-1][device_timing_key] = list()

    _ping_telemetry = False
    if ip:
        telemetry_thread.join()

    log("Saving results")
    results.save()
    # Make json files readable
    with open(results.json_name) as f:
        res = json.load(f)
    with open(results.json_name, "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    log.configure("training.log", print_level=Levels.DEBUG)  # Hydra controls cwd
    main()
