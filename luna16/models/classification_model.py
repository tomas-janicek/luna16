import numpy as np
import torch
from mlflow.models import infer_signature
from mlflow.pytorch import ModelSignature
from torch import nn
from torch.utils import data as data_utils

from luna16 import batch_iterators, message_handler, modules, utils

from .. import dto, enums
from . import base


class NoduleClassificationModel(base.BaseModel[dto.LunaClassificationCandidate]):
    def __init__(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_iterator: batch_iterators.BatchIteratorProvider,
        logger: message_handler.MessageHandler,
        log_every_n_examples: int,
        validation_cadence: int = 5,
    ) -> None:
        self.device, n_gpu_devices = utils.get_device()
        self.module = module
        if n_gpu_devices > 1:
            self.module = nn.DataParallel(module=self.module)
        self.module = self.module.to(self.device)
        self.optimizer = optimizer
        self.validation_cadence = validation_cadence
        self.batch_iterator = batch_iterator
        self.logger = logger
        self.log_every_n_examples = log_every_n_examples

    def prepare_for_fine_tuning_head(self) -> None:
        # Replace only luna head. Starting from a fully
        # initialized model would have us begin with (almost) all nodules
        # labeled as malignant, because that output means “nodule” in the
        # classifier we start from.
        self.module.luna_head = modules.LunaHead()

        finetune_blocks = ("luna_head",)

        # We to gather gradient only for blocks we will be fine-tuning.
        # This results in training only modifying parameters of finetune blocks.
        for name, parameter in self.module.named_parameters():
            if name.split(".")[0] not in finetune_blocks:
                parameter.requires_grad_(False)

    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[dto.LunaClassificationCandidate],
        validation_dl: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> dto.Scores:
        score = self.do_training(epoch=epoch, train_dataloader=train_dl)

        if epoch == 1 or epoch % self.validation_cadence:
            score = self.do_validation(epoch=epoch, validation_dataloader=validation_dl)

        return {"score": score}

    def do_training(
        self,
        epoch: int,
        train_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> np.float32:
        self.module.train()
        dataset_length = len(train_dataloader.dataset)  # type: ignore
        batch_metrics = dto.ClassificationBatchMetrics.create_empty(
            dataset_len=dataset_length, device=self.device
        )

        batch_iter = self.batch_iterator.enumerate_batches(
            train_dataloader,
            epoch=epoch,
            mode=enums.Mode.TRAINING,
            candidate_batch_type=dto.LunaClassificationCandidateBatch,
        )
        n_logged = 0
        score = np.float32(0.0)
        n_processed_training_samples = (epoch - 1) * dataset_length
        for _batch_index, batch in batch_iter:
            self.optimizer.zero_grad()

            loss, batch_metrics = self.compute_batch_loss(
                batch=batch,
                batch_metrics=batch_metrics,
            )

            loss.backward()
            self.optimizer.step()
            if n_processed_training_samples > self.log_every_n_examples * n_logged:
                score = self.log_metrics(
                    epoch=epoch,
                    n_processed_training_samples=n_processed_training_samples,
                    mode=enums.Mode.TRAINING,
                    metrics=batch_metrics,
                )
                n_logged += 1
            n_processed_training_samples += len(batch.candidate)

        score = self.log_metrics(
            epoch=epoch,
            n_processed_training_samples=n_processed_training_samples,
            mode=enums.Mode.TRAINING,
            metrics=batch_metrics,
        )
        return score

    def do_validation(
        self,
        epoch: int,
        validation_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> np.float32:
        with torch.no_grad():
            self.module.eval()
            dataset_length = len(validation_dataloader.dataset)  # type: ignore
            batch_metrics = dto.ClassificationBatchMetrics.create_empty(
                dataset_len=dataset_length, device=self.device
            )

            batch_iter = self.batch_iterator.enumerate_batches(
                validation_dataloader,
                epoch=epoch,
                mode=enums.Mode.VALIDATING,
                candidate_batch_type=dto.LunaClassificationCandidateBatch,
            )
            n_logged = 0
            score = np.float32(0.0)
            n_processed_training_samples = (epoch - 1) * dataset_length
            for _batch_index, batch in batch_iter:
                _, batch_metrics = self.compute_batch_loss(
                    batch=batch,
                    batch_metrics=batch_metrics,
                )
                if n_processed_training_samples > self.log_every_n_examples * n_logged:
                    score = self.log_metrics(
                        epoch=epoch,
                        n_processed_training_samples=n_processed_training_samples,
                        mode=enums.Mode.VALIDATING,
                        metrics=batch_metrics,
                    )
                    n_logged += 1
                n_processed_training_samples += len(batch.candidate)
            score = self.log_metrics(
                epoch=epoch,
                n_processed_training_samples=n_processed_training_samples,
                mode=enums.Mode.VALIDATING,
                metrics=batch_metrics,
            )
        return score

    def get_module(self) -> torch.nn.Module:
        return self.module

    def compute_batch_loss(
        self,
        batch: dto.LunaClassificationCandidateBatch,
        batch_metrics: dto.ClassificationBatchMetrics,
    ) -> tuple[torch.Tensor, dto.ClassificationBatchMetrics]:
        input = batch.candidate.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)

        logits, probability = self.module(input)

        cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        # Because redaction is set to `none`, we get tensor not single value
        # Target is list of 1 or 0 representing whether image is nodule.
        # Input parameter is list of list with two values. First is probability
        # that input will be 0 (not nodule) and second is probability that input
        # is 1 (is nodule).
        true_labels = labels[:, 1]
        true_probability = probability[:, 1]
        loss: torch.Tensor = cross_entropy_loss(input=logits, target=true_labels)

        batch_metrics.add_batch_metrics(
            loss=loss, labels=true_labels, predictions=true_probability
        )

        return loss.mean(), batch_metrics

    def get_signature(
        self, train_dl: data_utils.DataLoader[dto.LunaClassificationCandidate]
    ) -> ModelSignature:
        input = torch.unsqueeze(train_dl.dataset[0].candidate, 0)
        input = input.to(self.device, non_blocking=True)
        _logits, probability = self.module(input)
        signature = infer_signature(
            model_input=input.to("cpu").detach().numpy(),
            model_output=probability.to("cpu").detach().numpy(),
        )
        return signature

    def log_metrics(
        self,
        epoch: int,
        n_processed_training_samples: int,
        mode: enums.Mode,
        metrics: dto.ClassificationBatchMetrics,
        classification_threshold: float = 0.5,
    ) -> np.floating:
        negative_label_mask = metrics.labels == 0
        negative_prediction_mask = metrics.predictions <= classification_threshold

        positive_label_mask = metrics.labels == 1
        positive_prediction_mask = ~negative_prediction_mask

        negative_count = int(negative_label_mask.sum())
        positive_count = int(positive_label_mask.sum())

        true_negative_count = int(
            (negative_label_mask & negative_prediction_mask).sum()
        )
        true_positive_count = int(
            (positive_label_mask & positive_prediction_mask).sum()
        )

        false_positive_count = negative_count - true_negative_count
        false_negative_count = positive_count - true_positive_count

        epoch_metric: dict[str, dto.NumberValue] = {}
        loss = metrics.loss.mean().item()
        epoch_metric["loss/all"] = dto.NumberValue(
            name="Loss", value=loss, formatted_value=f"{loss:-5.4f}"
        )

        loss_negative = metrics.loss[negative_label_mask].mean().item()
        epoch_metric["loss/neg"] = dto.NumberValue(
            name="Loss Negative",
            value=loss_negative,
            formatted_value=f"{loss_negative:-5.4f}",
        )

        loss_positive = metrics.loss[positive_label_mask].mean().item()
        epoch_metric["loss/pos"] = dto.NumberValue(
            name="Loss Positive",
            value=loss_positive,
            formatted_value=f"{loss_positive:-5.4f}",
        )

        correct_all = (true_positive_count + true_negative_count) / np.float32(
            metrics.dataset_length()
        )
        epoch_metric["correct/all"] = dto.NumberValue(
            name="Correct All", value=correct_all, formatted_value=f"{correct_all:.0%}"
        )
        correct_negative = true_negative_count / np.float32(negative_count)
        epoch_metric["correct/neg"] = dto.NumberValue(
            name="Correct Negative",
            value=correct_negative,
            formatted_value=f"{correct_negative:.0%}",
        )

        correct_positive = true_positive_count / np.float32(positive_count)
        epoch_metric["correct/pos"] = dto.NumberValue(
            name="Correct Positive",
            value=correct_positive,
            formatted_value=f"{correct_positive:.0%}",
        )

        precision = true_positive_count / np.float32(
            true_positive_count + false_positive_count
        )
        recall = true_positive_count / np.float32(
            true_positive_count + false_negative_count
        )
        f1_score = 2 * (precision * recall) / (precision + recall)

        epoch_metric["pr/precision"] = dto.NumberValue(
            name="Precision", value=precision, formatted_value=f"{precision:-5.4f}"
        )
        epoch_metric["pr/recall"] = dto.NumberValue(
            name="Recall", value=recall, formatted_value=f"{recall:-5.4f}"
        )
        epoch_metric["pr/f1_score"] = dto.NumberValue(
            name="F1 Score", value=f1_score, formatted_value=f"{f1_score:-5.4f}"
        )

        log_metrics = message_handler.LogMetrics(
            epoch=epoch,
            mode=mode,
            n_processed_samples=n_processed_training_samples,
            values=epoch_metric,
        )
        self.logger.handle_message(log_metrics)

        log_results = message_handler.LogResult(
            epoch=epoch,
            mode=mode,
            n_processed_samples=n_processed_training_samples,
            predictions=metrics.predictions,
            labels=metrics.labels,
        )
        self.logger.handle_message(log_results)

        return f1_score

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"model={self.module.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"validation_cadence={self.validation_cadence}, "
            f"batch_iterator={self.batch_iterator.__class__.__name__}, "
            f"logger={self.logger.__class__.__name__})"
        )
        return _repr
