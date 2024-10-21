import hashlib
import logging
import shutil
import typing
from datetime import datetime

import torch
import torch.optim
from torch import nn

from luna16 import settings

_log = logging.getLogger(__name__)


ModuleT = typing.TypeVar("ModuleT", bound=nn.Module)


class ModelSaver:
    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.model_dir = settings.MODELS_DIR / self.model_name

    def save_model(
        self,
        *,
        epoch: int,
        n_processed_samples: int,
        training_started_at: str,
        modules: typing.Mapping[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        is_best: bool = False,
    ) -> None:
        models_path = (
            self.model_dir
            / f"{self.model_name}_{training_started_at}_{n_processed_samples}.state"
        )

        # We are saving only model's state instead of saving whole model.
        # This is because this gives us flexibility of loading this state
        # to different models than our UNetNormalized model.
        module_states = {}
        for module_name, module in modules.items():
            module_states[module_name] = str(module)
            module_states[f"{module_name}_state"] = module.state_dict()
            module_states[f"{module_name}_type"] = type(module).__name__
        state = {
            "time": datetime.now().isoformat(),
            # By saving the optimizer state as well, we could resume training seamlessly.
            "optimizer_name": type(optimizer).__name__,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "n_training_samples_processed": n_processed_samples,
            "modules": module_states,
        }

        # Instructions on how to load this model and optimizer to resume training can
        # be found here https://pytorch.org/tutorials/beginner/saving_loading_models.html.
        torch.save(state, models_path)

        _log.debug(f"Saved model params to {models_path}")

        if is_best:
            best_models_path = (
                self.model_dir / f"{self.model_name}_{training_started_at}.best.state"
            )
            shutil.copyfile(models_path, best_models_path)

            _log.debug(f"Saved model params to {best_models_path}")

        with open(models_path, "rb") as f:
            _log.debug("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def load_model(
        self, *, n_excluded_blocks: int, model: ModuleT, state_name: str
    ) -> ModuleT:
        state_path = self.model_dir / state_name

        training_state: typing.Mapping[str, typing.Any] = torch.load(
            state_path, map_location="cpu"
        )
        model_blocks = [
            name
            for name, submodule in model.named_children()
            if len(list(submodule.parameters())) > 0
        ]
        # Exclude `n_excluded_blocks` from model's blocks. This created `finetune_blocks`.
        # This block will be finetuned.
        finetune_blocks = model_blocks[-n_excluded_blocks:]
        _log.info(
            f"Fine-tuning from {state_path}, blocks: {', '.join(finetune_blocks)}"
        )
        # Create new state dict by taking everything from loaded state dict
        # except last block (the final linear part). Starting from a fully
        # initialized model would have us begin with (almost) all nodules
        # labeled as malignant, because that output means “nodule” in the
        # classifier we start from.
        state_dict = {
            k: v
            for k, v in training_state["modules"]["model_state"].items()
            if k.split(".")[0] not in model_blocks[-1]
        }

        model.load_state_dict(
            state_dict=state_dict,
            # Strict is set to False because we are not providing state dict with the same
            # dimensions as original state dict.
            strict=False,
        )
        # We to gather gradient only for blocks we will be fine-tuning.
        # This results in training only modifying parameters of finetune blocks.
        for name, parameter in model.named_parameters():
            if name.split(".")[0] not in finetune_blocks:
                parameter.requires_grad_(False)

        return model
