from __future__ import annotations
import base64
import io
import platform
import socket
from collections import OrderedDict

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return torch.load(buffer, map_location=device)

def is_rpi() -> bool:
    """ I'd just like to interject for a moment. What you're refering to as Linux, is in fact, GNU/Linux,
    or as I've recently taken to calling it, GNU plus Linux. Linux is not an operating system unto itself,
    but rather another free component of a fully functioning GNU system made useful by the GNU corelibs,
    shell utilities and vital system components comprising a full OS as defined by POSIX. """
    return all(kw in platform.platform() for kw in ("Linux", "aarch"))
