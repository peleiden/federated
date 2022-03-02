from __future__ import annotations
from threading import Thread
from typing import Any, Callable
import json
import functools
import os
import shlex
import socket
import subprocess
import time

from flask import Flask, jsonify, request
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, get_repo
import psutil

# Do not import from src, as flask will not be able to resolve imports
from client_utils import set_hostname, set_static_ip, get_ip

# Configure logging
os.makedirs("logs", exist_ok=True)
logpath = f"logs/{socket.gethostname()}.log"
log.configure(logpath, append=True)

# Create client server
client = Flask(__name__)
Api(client)
CORS(client)

def _delayed_reboot(seconds=1):
    def reboot():
        log(f"Rebooting in {seconds} seconds")
        time.sleep(seconds)
        log("Rebooting now")
        os.system("reboot")
    t = Thread(target=reboot)
    t.start()

def _get_post_data() -> dict[str, Any]:
    """ Returns data from a post request. Assumes json """
    # Return a dict parsed from json if possible
    if request.form:
        return request.form.to_dict()
    # Else parse raw data directly
    return json.loads(request.data.decode("utf-8"))

def _endpoint(fun: Callable):
    """ Used for annotating endpoint functions. This method ensures error handling
    and guarantees that all returns are of the format
    {
        "data": whatever or null if error,
        "error-message": str or null if no error
    } """
    @functools.wraps(fun)
    def fun_wrapper():
        log("Executing API endpoint %s" % fun.__name__)
        try:
            return_value = {
                "data": fun(),
                "error-message": None,
            }
            log("Returning %s" % return_value)
            return jsonify(return_value)
        except Exception as e:
            log.log_with_stacktrace(e)
            return jsonify({
                "data": None,
                "error-message": str(e),
            })
    return fun_wrapper

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
    """ Issue a list of system commands to the Pi.
    Expects the json to be a list of strings. """
    cmds = _get_post_data()
    for cmd in cmds:
        if cmd == "reboot":
            _delayed_reboot()
            return
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        code = p.wait()
        if code:
            raise RuntimeError(
                f"'{cmd}' gave error code {code}\n"
                f"stdout: {p.stdout.read().decode('utf-8')}\n"
                f"stderr: {p.stderr.read().decode('utf-8')}"
            )

@client.get("/configure")
@_endpoint
def configure():
    num = int(request.args.get("num"))
    set_hostname(f"SSR{num}")
    set_static_ip(f"192.168.0.{200+num}")
    _delayed_reboot()

if __name__ == "__main__":
    hostname = socket.gethostname()
    if hostname.startswith("SSR"):
        port = 3080 + int(hostname[3:])
    else:
        port = 3080
    client.run(host="0.0.0.0", port=port, debug=False)
