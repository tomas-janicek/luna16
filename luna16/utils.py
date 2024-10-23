import datetime

import torch

CUDA_DEVICE = torch.device("cuda")
MPS_DEVICE = torch.device("mps")
CPU_DEVICE = torch.device("cpu")


def get_device() -> tuple[torch.device, int]:
    """Function return pytorch device that can be used
    for calculation in current running system and number of
    GPU devices it can use.

    Returns:
        tuple[torch.device, int]: Tuple consisting from torch.device and number of available GPUs
    """
    if torch.cuda.is_available():
        # `torch.cuda` device enables high-performance training on GPU
        # Nvidia GPU.
        return torch.device("cuda"), torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        # `torch.backends.mps` device enables high-performance training on GPU
        # for MacOS devices with Metal programming framework.
        return torch.device("mps"), 1
    else:
        return torch.device("cpu"), 1


def get_datetime_string(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def get_model_state_name(name: str, version: str) -> str:
    return f"{name}_{version}.state"
