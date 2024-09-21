import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils import data as data_utils

from luna16 import batch_iterators, training_logging

from .. import dto, enums, modules, utils
from . import base

_log = logging.getLogger(__name__)


_cuda_device = torch.device("cuda")


class LunaTrainingAPI(base.BaseTrainingAPI):
    def __init__(
        self,
        *,
        model: modules.LunaModel,
        optimizer: torch.optim.Optimizer,
        num_workers: int,
        batch_size: int,
        training_name: str,
        classification_logger: training_logging.ClassificationLoggingAdapter,
        batch_iterator: batch_iterators.BaseIteratorProvider,
        model_saver: base.ModelSaver | None = None,
    ) -> None:
        self.training_name = training_name
        self.model: modules.LunaModel | nn.DataParallel[modules.LunaModel] = model
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_processed_training_samples = 0
        self.classification_logger = classification_logger
        self.training_started_at = datetime.now()
        self.batch_iterator = batch_iterator
        self.model_saver = model_saver

        # Initialize model on GPU if available
        self.device, _ = utils.get_device()
        self.n_gpu_device = 1
        if self.is_using_cuda:
            self.n_gpu_device = torch.cuda.device_count()
            _log.info(f"Using CUDA; {self.n_gpu_device} devices.")
            if self.n_gpu_device > 1:
                self.model = nn.DataParallel(module=self.model)
        self.model = self.model.to(self.device)
        self.modules = {"model": self.model}

    @property
    def is_using_cuda(self) -> bool:
        return self.device == _cuda_device

    @classmethod
    def create_with_optimizer_and_model(
        cls,
        num_workers: int,
        batch_size: int,
        training_name: str,
        model_name: str,
        classification_logger: training_logging.ClassificationLoggingAdapter,
    ) -> "LunaTrainingAPI":
        model = modules.LunaModel()
        model_saver = base.ModelSaver(model_name=model_name)
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.99)
        batch_iterator = batch_iterators.BatchIteratorProvider(
            batch_loggers=classification_logger.batch_loggers
        )
        return cls(
            num_workers=num_workers,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
            training_name=training_name,
            classification_logger=classification_logger,
            batch_iterator=batch_iterator,
            model_saver=model_saver,
        )

    @classmethod
    def create_from_saved_state(
        cls,
        num_workers: int,
        batch_size: int,
        training_name: str,
        state_name: str,
        model_name: str,
        loaded_model_name: str,
        classification_logger: training_logging.ClassificationLoggingAdapter,
    ) -> "LunaTrainingAPI":
        model = modules.LunaModel()
        classification_model_saver = base.ModelSaver(model_name=loaded_model_name)
        model = classification_model_saver.load_model(
            model=model, state_name=state_name, n_excluded_blocks=2
        )
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.99)
        batch_iterator = batch_iterators.BatchIteratorProvider(
            batch_loggers=classification_logger.batch_loggers
        )
        malignancy_model_saver = base.ModelSaver(model_name=model_name)
        return cls(
            num_workers=num_workers,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
            training_name=training_name,
            classification_logger=classification_logger,
            batch_iterator=batch_iterator,
            model_saver=malignancy_model_saver,
        )

    def start_training(
        self,
        *,
        epochs: int,
        train: data_utils.Dataset[dto.LunaClassificationCandidate],
        validation: data_utils.Dataset[dto.LunaClassificationCandidate],
        validation_cadence: int = 5,
    ) -> None:
        train_dl = self.get_training_dataloader(train=train)
        validation_dl = self.get_validation_dataloader(validation=validation)

        self.classification_logger.log_start_training(
            training_api=self,
            n_epochs=epochs,
            batch_size=self.batch_size,
            train_dl=train_dl,
            validation_dl=validation_dl,
        )

        for epoch in range(1, epochs + 1):
            self.classification_logger.log_epoch(epoch=epoch)

            training_metrics = self.do_training(epoch, train_dl)
            self.log_metrics(
                epoch=epoch, mode=enums.Mode.TRAINING, metrics=training_metrics
            )

            best_score = 0.0
            if epoch == 1 or epoch % validation_cadence:
                validation_metrics = self.do_validation(epoch, validation_dl)
                score = self.log_metrics(
                    epoch=epoch, mode=enums.Mode.VALIDATING, metrics=validation_metrics
                )
                best_score = max(score, best_score)

                if self.model_saver:
                    self.model_saver.save_model(
                        epoch=epoch,
                        is_best=score == best_score,
                        n_processed_samples=self.n_processed_training_samples,
                        training_started_at=self.training_started_at.isoformat(),
                        modules=self.modules,
                        optimizer=self.optimizer,
                    )

    def do_training(
        self,
        epoch: int,
        train_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> dto.ClassificationBatchMetrics:
        self.model.train()
        dataset_length = len(train_dataloader.dataset)  # type: ignore
        batch_metrics = dto.ClassificationBatchMetrics.create_empty(
            dataset_len=dataset_length, device=self.device
        )

        batch_iter = self.batch_iterator.enumerate_batches(
            train_dataloader, epoch=epoch, mode=enums.Mode.TRAINING
        )
        for _batch_index, batch in batch_iter:
            self.optimizer.zero_grad()

            loss, batch_metrics = self.compute_batch_loss(
                batch=batch,
                batch_metrics=batch_metrics,
            )

            loss.backward()
            self.optimizer.step()

        self.n_processed_training_samples += dataset_length

        return batch_metrics

    def do_validation(
        self,
        epoch: int,
        validation_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> dto.ClassificationBatchMetrics:
        with torch.no_grad():
            self.model.eval()
            dataset_length = len(validation_dataloader.dataset)  # type: ignore
            batch_metrics = dto.ClassificationBatchMetrics.create_empty(
                dataset_len=dataset_length, device=self.device
            )

            batch_iter = self.batch_iterator.enumerate_batches(
                validation_dataloader, epoch=epoch, mode=enums.Mode.VALIDATING
            )
            for _batch_index, batch in batch_iter:
                _, batch_metrics = self.compute_batch_loss(
                    batch=batch,
                    batch_metrics=batch_metrics,
                )

        return batch_metrics

    def compute_batch_loss(
        self,
        batch: dto.LunaClassificationCandidate,
        batch_metrics: dto.ClassificationBatchMetrics,
    ) -> tuple[torch.Tensor, dto.ClassificationBatchMetrics]:
        input = batch.candidate.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)

        logits, probability = self.model(input)

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

    def get_training_dataloader(
        self, train: data_utils.Dataset[dto.LunaClassificationCandidate]
    ) -> data_utils.DataLoader[dto.LunaClassificationCandidate]:
        batch_size = self.batch_size * self.n_gpu_device
        train_dataloader = data_utils.DataLoader(
            dataset=train,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.is_using_cuda,
        )
        return train_dataloader

    def get_validation_dataloader(
        self, validation: data_utils.Dataset[dto.LunaClassificationCandidate]
    ) -> data_utils.DataLoader[dto.LunaClassificationCandidate]:
        batch_size = self.batch_size * self.n_gpu_device

        validation_dataloader = data_utils.DataLoader(
            dataset=validation,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.is_using_cuda,
        )
        return validation_dataloader

    def log_metrics(
        self,
        epoch: int,
        mode: enums.Mode,
        metrics: dto.ClassificationBatchMetrics,
        classification_threshold: float = 0.5,
    ) -> np.float32:
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

        epoch_metric: dict[str, training_logging.NumberValue] = {}
        loss = metrics.loss.mean().item()
        epoch_metric["loss/all"] = training_logging.NumberValue(
            name="Loss", value=loss, formatted_value=f"{loss:-5.4f}"
        )

        loss_negative = metrics.loss[negative_label_mask].mean().item()
        epoch_metric["loss/neg"] = training_logging.NumberValue(
            name="Loss Negative",
            value=loss_negative,
            formatted_value=f"{loss_negative:-5.4f}",
        )

        loss_positive = metrics.loss[positive_label_mask].mean().item()
        epoch_metric["loss/pos"] = training_logging.NumberValue(
            name="Loss Positive",
            value=loss_positive,
            formatted_value=f"{loss_positive:-5.4f}",
        )

        correct_all = (true_positive_count + true_negative_count) / np.float32(
            metrics.dataset_length()
        )
        epoch_metric["correct/all"] = training_logging.NumberValue(
            name="Correct All", value=correct_all, formatted_value=f"{correct_all:.0%}"
        )
        correct_negative = true_negative_count / np.float32(negative_count)
        epoch_metric["correct/neg"] = training_logging.NumberValue(
            name="Correct Negative",
            value=correct_negative,
            formatted_value=f"{correct_negative:.0%}",
        )

        correct_positive = true_positive_count / np.float32(positive_count)
        epoch_metric["correct/pos"] = training_logging.NumberValue(
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

        epoch_metric["pr/precision"] = training_logging.NumberValue(
            name="Precision", value=precision, formatted_value=f"{precision:-5.4f}"
        )
        epoch_metric["pr/recall"] = training_logging.NumberValue(
            name="Recall", value=recall, formatted_value=f"{recall:-5.4f}"
        )
        epoch_metric["pr/f1_score"] = training_logging.NumberValue(
            name="F1 Score", value=f1_score, formatted_value=f"{f1_score:-5.4f}"
        )

        self.classification_logger.log_metrics(
            epoch=epoch,
            mode=mode,
            n_processed_samples=self.n_processed_training_samples,
            values=epoch_metric,
        )

        self.classification_logger.log_results(
            epoch=epoch,
            mode=mode,
            n_processed_samples=self.n_processed_training_samples,
            predictions=metrics.predictions,
            labels=metrics.labels,
        )

        return f1_score

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"device={self.device}, "
            f"n_gpu_device={self.n_gpu_device}, "
            f"num_workers={self.num_workers}, "
            f"batch_size={self.batch_size})"
        )
