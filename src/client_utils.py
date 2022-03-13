import base64
import io
import socket
from collections import OrderedDict

import torch


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def state_dict_to_base64(state_dict: OrderedDict) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read())
    return b64.decode("ascii")

def state_dict_from_base64(b64: str) -> OrderedDict:
    buffer = io.BytesIO(base64.b64decode(b64))
    return torch.load(buffer)
