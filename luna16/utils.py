import torch


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
