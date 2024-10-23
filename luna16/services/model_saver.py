import logging
import typing

import torch
import torch.optim
from torch import nn

from luna16 import settings

_log = logging.getLogger(__name__)

ModuleT = typing.TypeVar("ModuleT", bound=nn.Module)


class BaseModelSaver(typing.Protocol):
    def save_model(self, *, name: str, module: nn.Module, version: str) -> None: ...

    def load_model(
        self, *, name: str, module: nn.Module, version: str
    ) -> nn.Module: ...


class ModelSaver(BaseModelSaver):
    def __init__(self) -> None:
        self.models_dir = settings.MODELS_DIR

    def save_model(self, *, name: str, module: nn.Module, version: str) -> None:
        model_dir = self.models_dir / name.lower()
        model_dir.mkdir(exist_ok=True, parents=True)
        models_path = model_dir / self.get_model_state_name(name, version)

        # We are saving only model's state instead of saving whole model.
        # This is because this gives us flexibility of loading this state
        # to different models.
        state = module.state_dict()

        # Instructions on how to load this model and optimizer to resume training can
        # be found here https://pytorch.org/tutorials/beginner/saving_loading_models.html.
        torch.save(state, models_path)

        _log.debug(f"Saved model params to {models_path}")

    def load_model(self, *, name: str, module: ModuleT, version: str) -> ModuleT:
        state_dict = self.load_state_dict(name, version)
        module.load_state_dict(
            state_dict=state_dict,
            # Strict is set to False because we are not providing state dict with the same
            # dimensions as original state dict.
            strict=False,
        )
        return module

    def load_state_dict(
        self, name: str, version: str
    ) -> typing.Mapping[str, typing.Any]:
        state_path = self.models_dir / name / self.get_model_state_name(name, version)
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        return state_dict

    @staticmethod
    def get_model_state_name(name: str, version: str) -> str:
        return f"{name.lower()}_{version}.state"
