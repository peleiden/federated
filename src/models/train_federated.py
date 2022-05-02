from __future__ import annotations

import json
import os
import time
import traceback as tb
from dataclasses import dataclass
from threading import Thread
from typing import Generator, Optional, OrderedDict

import hydra
import numpy as np
import requests
from omegaconf.dictconfig import DictConfig
from dotenv import load_dotenv
from pelutils import DataStorage, Levels, TickTock, log, TT

import wandb
from src.client_utils import state_dict_from_base64, state_dict_to_base64
from src.models.client_train import ClientTrainer
from src.models.server_train import ServerTrainer
from src.models.train_model import evaluate


_ping_telemetry = True
_timeout = 180

# Use WANDB except if WANDB_OFF is set
USE_WANDB = "WANDB_OFF" not in os.environ

@dataclass
class Results(DataStorage):
    ignore_missing=True
    # Training configuration
    cfg: dict
    start_args: dict
    clients: int
    clients_per_round: int
    num_images: int
    train_time: float
    max_time: float
    comm_rounds: int
    pct_noisy_images_by_round: list[float]
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

    # [ comm round [ device [ value ] ] ]
    local_accs: list[list[list[float]]]
    local_losses: list[list[list[float]]]
    local_train_accs: list[list[list[float]]]

    ip: str | None

    json_name = "results.json"


def dict_config_to_dict(dc: DictConfig) -> dict:
    d = dict()
    for kw, v in dc.items():
        if isinstance(v, DictConfig):
            d[kw] = dict_config_to_dict(v)
        else:
            d[kw] = v
    return d

def setup_local_clients(start_args: dict, num_clients: int) -> tuple[ClientTrainer]:
    return tuple(ClientTrainer(**start_args) for _ in range(num_clients))

def run_local_rounds(
    clients: tuple[ClientTrainer], client_args: list
) -> Generator[tuple, None, None]:
    for i, args in enumerate(client_args):
        yield clients[i].run_round(**args)

def connect_devices(ips: list[str], num_devices: int):
    """ Ensure that all devices respond. Two minute response time is allowed,
    as devices restart between sessions. """
    timer = TickTock()
    timer.tick()

    def connect_single_device(num: int):
        log("Connecting to device %i" % num)
        while timer.tock() < _timeout:
            try:
                response = requests.get(f"http://{ips[num]}:{3080+num}/ping")
                log.debug("Got status code %i from device %i" % (response.status_code, num))
                if response.status_code != 200:
                    raise IOError(
                        "Device %i returned status code %i" % (num, response.status_code)
                    )
                break
            except OSError:
                continue

    threads = list()
    for i in range(num_devices):
        threads.append(Thread(target=lambda: connect_single_device(i)))
        threads[-1].start()
    for i in range(num_devices):
        threads[i].join()

def setup_external_clients(ips: list[str], start_args: dict, num_devices: int, training_id: int) -> list[float]:

    def setup_single_client(num: int):
        log("Sending training configuration to device %i" % num)
        response = requests.post(f"http://{ips[num]}:{3080+num}/configure-training", json=dict(
            train_cfg=dict_config_to_dict(start_args["train_cfg"]),
            model_cfg=dict_config_to_dict(start_args["model_cfg"]),
            training_id=training_id,
        ))
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i\nError: %s" % (num, response.status_code, response.content))
        log("Device %i configured" % num)

    threads = list()
    for i in range(num_devices):
        threads.append(Thread(target=lambda: setup_single_client(i)))
        threads[-1].start()
    for i in range(num_devices):
        threads[i].join()

def run_external_rounds(ips: list[str], client_args: list, training_id: int) -> Generator[tuple[OrderedDict, dict[str, float], list[list[float]], list[list[float]]], None, None]:
    returned_b64s: list[str] = [None] * len(client_args)
    returned_timings: list[dict[str, float]] = [None] * len(client_args)
    returned_accs: list[list[float]] = [None] * len(client_args)
    returned_losses: list[list[float]] = [None] * len(client_args)
    returned_train_accs: list[list[float]] = [None] * len(client_args)

    def train_single_client(num: int, args: dict):
        args = args.copy()
        args["state_dict"] = state_dict_to_base64(args["state_dict"])
        args["training_id"] = training_id
        log("Sending state dict to device %i along with %i data indices" % (num, len(args["split"])))
        tt = TickTock()
        tt.tick()
        response = requests.post(f"http://{ips[num]}:{3080+num}/train-round", json=args)
        response_time = tt.tock()
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i\nError: %s" % (num, response.status_code, response.content))
        log("Device %i finished local training" % num)
        response = json.loads(response.content)
        returned_b64s[num] = response["data"]["state_dict"]
        returned_timings[num] = response["data"]["timings"]
        returned_timings[num]["response_time"] = response_time
        returned_accs[num] = response["data"]["accs"]
        returned_losses[num] = response["data"]["losses"]
        returned_train_accs[num] = response["data"]["train_accs"]

    threads = list()
    for i, args in enumerate(client_args):
        threads.append(Thread(target=lambda: train_single_client(i, args)))
        threads[-1].start()
    for i in range(len(client_args)):
        threads[i].join()
        yield state_dict_from_base64(returned_b64s[i]), returned_timings[i],\
            returned_accs[i], returned_losses[i], returned_train_accs[i]

def ping_telemetry(ips: list[str], num_clients: int, results: Results):
    global _ping_telemetry
    current_client = 0
    while _ping_telemetry:
        log.debug("Requesting telemetry from client %i" % current_client)
        try:
            response = requests.get(f"http://{ips[current_client]}:{3080+current_client}/telemetry")
            if response.status_code != 200:
                raise IOError("Device %i returned status code %i\nError: %s" % (current_client, response.status_code, response.content))
        except:
            log("Failed to get telemetry from device %i" % current_client, "Stacktrace", tb.format_exc())
            current_client = (current_client + 1) % num_clients
            time.sleep(0.1)
            continue
        results.telemetry[current_client]["timestamp"].append(time.time())
        results.telemetry[current_client]["memory_usage"].append(json.loads(response.content)["data"]["total-memory-usage-pct"])
        log.debug("Device %i reported %.2f %% memory usage" % (current_client, results.telemetry[current_client]["memory_usage"][-1]))
        current_client = (current_client + 1) % num_clients
        time.sleep(0.1)

def reset_all_devices(ips: list[str], num_clients: int):
    if not ips:
        return

    def reset_device(num: int):
        response = requests.get(f"http://{ips[num]}:{3080+num}/end-training")
        if response.status_code != 200:
            raise IOError("Device %i returned status code %i\nError: %s" % (num, response.status_code, response.content))
        log("Reset device %i" % num)

    threads = list()
    for i in range(num_clients):
        threads.append(Thread(target=lambda: reset_device(i)))
        threads[-1].start()
    for i in range(num_clients):
        threads[i].join()

@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    global _ping_telemetry
    _ping_telemetry = True
    log("Using wandb: %s" % USE_WANDB)
    server = ServerTrainer(cfg)
    start_args = server.get_client_start_args()

    entity = os.getenv("WANDB_ENTITY")
    project = "federated"
    name = cfg.configs.name if "name" in cfg.configs.keys() else "Name not defined"
    # Expects a token in envs called WANDB_API_KEY.
    # This key should match the owner of the key.
    if USE_WANDB:
        wandb.init(project=project, entity=entity, name=name)

    tt = TickTock()

    ip: Optional[str] = os.environ.get("IP")

    max_time = server.train_cfg.max_time
    log("Stopping in %.2f s" % max_time)

    results = Results(
        cfg               = cfg,
        start_args        = start_args,
        clients           = server.train_cfg.clients,
        clients_per_round = server.train_cfg.clients_per_round,
        num_images        = 0,
        train_time        = 0,
        max_time          = max_time,
        comm_rounds       = 0,
        pct_noisy_images_by_round = list(),
        timings           = list(),
        telemetry         = [{"timestamp": list(), "memory_usage": list()} for _ in range(server.train_cfg.clients_per_round)],
        eval_timestamps   = list(),
        test_accuracies   = list(),
        test_losses       = list(),
        local_accs        = list(),
        local_losses      = list(),
        local_train_accs  = list(),
        ip                = ip,
    )
    if ip and ip.endswith("x"):
        start = int(ip.split(".")[-1][:-1])
        ips = [".".join(ip.split(".")[:-1]) + ".%i" % (start+i) for i in range(results.clients_per_round)]
    elif ip:
        ips = [ip] * results.clients_per_round
    else:
        ips = list()

    log.section("Setting up clients")
    if ip:
        log("Using clients at IP %s" % ip)
        timestamp = time.time()
        training_id = hash(timestamp)
        log("Training ID: %i" % training_id)
        connect_devices(ips, server.train_cfg.clients_per_round)
        log("Connected to all devices")
        telemetry_thread = Thread(target=lambda: ping_telemetry(ips, server.train_cfg.clients_per_round, results))
        telemetry_thread.start()
        setup_external_clients(ips, start_args, server.train_cfg.clients_per_round, training_id)
    else:
        log("Using local training")
        clients = setup_local_clients(start_args, server.train_cfg.clients)

    train_timer = TickTock()
    train_timer.tick()

    acc, loss = evaluate(
        server.model,
        server.device,
        server.test_dataloader,
        server.criterion,
        use_wandb=USE_WANDB,
    )
    results.eval_timestamps.append(time.time())
    results.test_accuracies.append(float(acc))
    results.test_losses.append(float(loss))

    for i in range(cfg.configs.training.communication_rounds):
        if train_timer.tock() > max_time:
            break
        tt.tick()
        client_args = server.get_communication_round_args()
        client_idcs = { c["idx"] for c in client_args }
        log.section(
            f"Round {i}. Chose clients with idx", client_idcs
        )
        pct_noisy_clients = 100 * np.array([idx in server.noisy_clients for idx in client_idcs]).mean()
        results.pct_noisy_images_by_round.append(pct_noisy_clients * server.train_cfg["noisy_images"])
        log("%.4f %% noisy data this round" % results.pct_noisy_images_by_round[-1])

        if ips:
            received_data, timings, local_accs, local_losses, local_train_losses = zip(*run_external_rounds(ips, client_args, training_id))
        else:
            received_data, local_accs, local_losses, local_train_losses = zip(*run_local_rounds(clients, client_args))
        server.aggregate(received_data)
        train_time = tt.tock()

        tt.tick()
        acc, loss = evaluate(
            server.model,
            server.device,
            server.test_dataloader,
            server.criterion,
            use_wandb=USE_WANDB,
        )
        test_time = tt.tock()

        results.eval_timestamps.append(time.time())
        results.test_accuracies.append(float(acc))
        results.test_losses.append(float(loss))
        results.local_accs.append(local_accs)
        results.local_losses.append(local_losses)
        results.local_train_accs.append(local_train_losses)
        results.timings.append(
            dict(
                total_train=train_time,
                total_test=test_time,
            )
        )
        for device_timing_key in "response_time", "decode", "train", "encode":
            if ips:
                results.timings[-1][device_timing_key] = [
                    t[device_timing_key] for t in timings
                ]
            else:
                results.timings[-1][device_timing_key] = list()

        results.comm_rounds += 1
        results.num_images += server.train_cfg.local_epochs *\
            server.train_cfg.clients_per_round * server.train_cfg.local_data_amount

    log.section("Cleaning up")
    results.train_time = train_timer.tock()
    log("Images per second: %.2f" % (results.num_images / results.train_time))

    _ping_telemetry = False
    if ips:
        telemetry_thread.join()
    # Reset all devices, so they are ready for another training
    log("Resetting devices")
    reset_all_devices(ips, server.train_cfg.clients_per_round)

    log("Saving results")
    results.save()
    # Make json files readable
    with open(results.json_name) as f:
        res = json.load(f)
    with open(results.json_name, "w") as f:
        json.dump(res, f, indent=4)

    # Stop wandb so multirun does not fail
    if USE_WANDB:
        wandb.finish()

    # Give devices time to shut down before next round
    time.sleep(10)


if __name__ == "__main__":
    # Loads .env file, if in same folder as
    log.configure("training.log")  # Hydra controls cwd
    if load_dotenv():
        log.info(".env file found")
    else:
        log.info(".env file not found")

    try:
        main()
    except:
        # Stop telemetry pinging if something goes wrong
        _ping_telemetry = False
