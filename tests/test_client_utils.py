from src.client_utils import state_dict_to_base64, state_dict_from_base64
from src.models.architectures.conv import SimpleConv

import torch


def validate_state_dicts(model_state_dict_1, model_state_dict_2) -> bool:
    # Adapted from here
    # https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
    if len(model_state_dict_1) != len(model_state_dict_2):
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(model_state_dict_1.items(), model_state_dict_2.items()):
        if k_1 != k_2:
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            return False

    return True

def test_base64():
    net = SimpleConv((1, 100, 100), 10, 16, 8, 3, 2, 50, 0.1, 0.1)
    b64 = state_dict_to_base64(net.state_dict())
    state_dict = state_dict_from_base64(b64)
    assert validate_state_dicts(net.state_dict(), state_dict)
