from __future__ import annotations

import functools
import gc
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback as tb
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

import psutil
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restful import Api
from pelutils import get_repo, log, TickTock, Levels

sys.path.append(str(Path(__file__).parent.parent))

from src.client_utils import get_ip, state_dict_from_base64, state_dict_to_base64, is_rpi
from src.models.client_train import ClientTrainer

# Create client server
client = Flask(__name__)
Api(client)
CORS(client)

training_id: Optional[int] = None
config: Optional[None] = None
trainer: Optional[ClientTrainer] = None


def _delayed_reboot(seconds=1):
    def reboot():
        log(f"Rebooting in {seconds} seconds")
        time.sleep(seconds)
        log("Rebooting now")
        os.system("sudo reboot")

    t = Thread(target=reboot)
    t.start()

def _get_post_data() -> dict[str, Any]:
    """Returns data from a post request. Assumes json"""
    # Return a dict parsed from json if possible
    if request.form:
        return request.form.to_dict()
    # Else parse raw data directly
    return json.loads(request.data.decode("utf-8"))

def _endpoint(fun: Callable):
    """Used for annotating endpoint functions. This method ensures error handling
    and guarantees that all returns are of the format
    {
        "data": whatever or null if error,
        "error-message": str or null if no error
    }"""

    @functools.wraps(fun)
    def fun_wrapper():
        log("Executing API endpoint %s" % fun.__name__)
        try:
            return_value = {
                "data": fun(),
                "error-message": None,
            }
            log("Successfully got return value")
            gc.collect()
            return jsonify(return_value)
        except Exception as e:
            log.error(tb.format_exc())
            return jsonify(
                {
                    "data": None,
                    "error-message": tb.format_exc(),
                }
            ), 500

    return fun_wrapper

def _reserve(received_training_id: int):
    """Used for annotating endpoint functions. This method ensures error handling
    and guarantees that all returns are of the format
    {
        "data": whatever or null if error,
        "error-message": str or null if no error
    }"""
    global training_id
    if training_id is not None and received_training_id != training_id:
        raise IOError("Client is already reserved for training")

@client.get("/ping")
@_endpoint
def ping():
    return {
        "hostname": socket.gethostname(),
        "local-ip": get_ip(),
        "commit": get_repo()[1],
    }

@client.get("/logs")
@_endpoint
def logs():
    with open(logpath) as logfile:
        return logfile.read()

@client.get("/telemetry")
@_endpoint
def telemetry():
    return {
        "timestamp": time.time(),
        "process-memory-usage": psutil.Process().memory_info().rss,
        "total-memory-usage-pct": psutil.virtual_memory().percent,
        "total-memory-usage": psutil.virtual_memory().used,
        "total-memory": psutil.virtual_memory().total,
        "cpu-freq": [x.current for x in psutil.cpu_freq(percpu=True)],
        "cpu-min-freq": psutil.cpu_freq().min,
        "cpu-max-freq": psutil.cpu_freq().max,
        "cpu-usage-pct": psutil.cpu_percent(0.1, percpu=True),
    }

@client.post("/command")
@_endpoint
def command():
    """Issue a list of system commands to the Pi.
    Expects the json to be a list of strings."""
    cmds = _get_post_data()
    log("Got commands:", *cmds)
    for cmd in cmds:
        log.debug("Command: %s" % cmd)
        if cmd == "reboot":
            _delayed_reboot()
            return
        p = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        code = p.wait()
        if code:
            raise RuntimeError(
                f"'{cmd}' gave error code {code}\n"
                f"stdout: {p.stdout.read().decode('utf-8')}\n"
                f"stderr: {p.stderr.read().decode('utf-8')}"
            )

@client.post("/configure-training")
@_endpoint
def configure_training():
    global training_id, trainer
    start_args = _get_post_data()
    received_training_id = start_args.pop("training_id")
    _reserve(received_training_id)
    training_id = received_training_id
    trainer = ClientTrainer(**start_args, data_path=os.path.abspath(os.path.join(__file__, "..", "..", "data")))

@client.post("/train-round")
@_endpoint
def train_round() -> str:
    """ Performs a training round. Expects a json of
    {
        "indices": list[int],
        "augmentation": int that tells what augmentation to do,
        "state_dict": base64 encoding of state_dict
    } """
    global trainer
    args = _get_post_data()
    _reserve(args.pop("training_id"))
    timings = dict()
    tt = TickTock()

    tt.tick()
    args["state_dict"] = state_dict_from_base64(args["state_dict"])
    timings["decode"] = tt.tock()

    tt.tick()
    state_dict, accs, losses, train_accs = trainer.run_round(**args)
    timings["train"] = tt.tock()

    tt.tick()
    b64 = state_dict_to_base64(state_dict)
    timings["encode"] = tt.tock()

    return {
        "state_dict": b64,
        "timings": timings,
        "accs": accs,
        "losses": losses,
        "train_accs": train_accs,
    }

@client.get("/end-training")
@_endpoint
def end_training():
    global training_id, trainer
    # Try to force garbage collection
    del training_id, trainer
    training_id = None
    trainer = None
    if is_rpi():
        # Prevent memory from piling up, as it is not properly cleared
        _delayed_reboot()

if __name__ == "__main__":
    hostname = socket.gethostname()
    if hostname.startswith("SSR"):
        port = 3080 + int(hostname[3:])
    else:
        port = os.environ.get("PORT", 3080)

    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logpath = f"logs/{socket.gethostname()}:{port}.log"
    log.configure(logpath, append=True, print_level=Levels.DEBUG)
    client.run(host="0.0.0.0", port=port, debug=False, processes=1, threaded=True)
