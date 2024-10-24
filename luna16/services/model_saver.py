import logging
import tempfile
import typing

import mlflow
import torch
import torch.optim
from mlflow.pytorch import ModelSignature
from torch import nn
from torchinfo import Verbosity, summary

from luna16 import settings

_log = logging.getLogger(__name__)

ModuleT = typing.TypeVar("ModuleT", bound=nn.Module)


class BaseModelSaver(typing.Protocol):
    def load_model(
        self, *, name: str, module_class: type[ModuleT], version: str
    ) -> ModuleT: ...


class ModelSaver(BaseModelSaver):
    def __init__(self) -> None:
        self.models_dir = settings.MODELS_DIR

    def save_model(self, *, name: str, module: nn.Module, version: str) -> None:
        model_dir = self.models_dir / name.lower()
        model_dir.mkdir(exist_ok=True, parents=True)
        state_name = self.get_model_state_name(name, version)
        models_path = model_dir / f"{state_name}.state"

        # We are saving only model's state instead of saving whole model.
        # This is because this gives us flexibility of loading this state
        # to different models.
        state = module.state_dict()

        # Instructions on how to load this model and optimizer to resume training can
        # be found here https://pytorch.org/tutorials/beginner/saving_loading_models.html.
        torch.save(state, models_path)

        summaries_path = self.models_dir / "summaries"
        summaries_path.mkdir(exist_ok=True, parents=True)
        model_summary_path = summaries_path / f"{state_name}_model_summary.txt"
        with open(model_summary_path, "w+") as f:
            # Verbosity quiet is set so model summary is not printed to stdout
            model_summary = summary(module, verbose=Verbosity.QUIET)
            f.write(str(model_summary))

        _log.debug(f"Saved model params to {models_path}.")

    def load_model(
        self, *, name: str, module_class: type[ModuleT], version: str
    ) -> ModuleT:
        state_dict = self.load_state_dict(name, version)
        module = module_class()
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
        state_path = (
            self.models_dir
            / name.lower()
            / f"{self.get_model_state_name(name, version)}.state"
        )
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        return state_dict

    @staticmethod
    def get_model_state_name(name: str, version: str) -> str:
        return f"{name.lower()}_{version}.state"


class MLFlowModelSaver(BaseModelSaver):
    def __init__(self) -> None:
        self.models_dir = settings.MODELS_DIR

    def save_model(
        self, *, name: str, module: nn.Module, signature: ModelSignature
    ) -> None:
        with tempfile.NamedTemporaryFile("w") as tmp:
            # Verbosity quiet is set so model summary is not printed to stdout
            model_summary = summary(module, verbose=Verbosity.QUIET)
            tmp.write(str(model_summary))
            mlflow.log_artifact(tmp.name)
        _model_info = mlflow.pytorch.log_model(
            pytorch_model=module,
            artifact_path=f"{name.lower()}_model",
            registered_model_name=name,
            signature=signature,
        )
        _log.debug(f"Saved model params to MLFLow under '{name}' name.")

    def load_model(
        self, *, name: str, module_class: type[ModuleT], version: str
    ) -> ModuleT:
        # Version used here is not the same version that we manually define (like in ModelSaver above).
        # This version is defined by MLFlow automatically. This process can not be overridden.
        model_uri = f"models:/{name}/{version}"
        module = mlflow.pytorch.load_model(model_uri=model_uri, map_location="cpu")
        if not isinstance(module, module_class):
            raise TypeError(
                f"Loaded module is type {module.__class__.__name__} but type {module_class.__name__} was requested."
            )
        return module

    def load_state_dict(
        self, name: str, version: str
    ) -> typing.Mapping[str, typing.Any]:
        model_uri = f"models:/{name}/{version}"
        state_dict = mlflow.pytorch.load_model(
            model_uri=model_uri,
            map_location="cpu",
            weights_only=True,
            pickle_module=None,
        )
        return state_dict
