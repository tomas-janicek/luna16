import numpy as np
import torch
from mlflow.models import infer_signature
from mlflow.pytorch import ModelSignature
from torch import nn
from torch.utils import data as data_utils

from luna16 import batch_iterators, dto, enums, message_handler, scoring, utils

from . import base


class NoduleClassificationModel(base.BaseModel[dto.LunaClassificationCandidate]):
    def __init__(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        self.lr_scheduler = lr_scheduler
        self.validation_cadence = validation_cadence
        self.batch_iterator = batch_iterator
        self.logger = logger
        self.log_every_n_examples = log_every_n_examples

    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[dto.LunaClassificationCandidate],
        validation_dl: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> scoring.PerformanceMetrics:
        score = self.do_training(epoch=epoch, train_dataloader=train_dl)

        if epoch == 1 or epoch % self.validation_cadence:
            score = self.do_validation(epoch=epoch, validation_dataloader=validation_dl)

        self.lr_scheduler.step()

        return score

    def do_training(
        self,
        epoch: int,
        train_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> scoring.PerformanceMetrics:
        self.module.train()
        dataset_length = len(train_dataloader.dataset)  # type: ignore
        epoch_metrics = scoring.ClassificationMetrics.create_empty(
            dataset_len=dataset_length, device=self.device
        )

        batch_iter = self.batch_iterator.enumerate_batches(
            train_dataloader,
            epoch=epoch,
            mode=enums.Mode.TRAINING,
            candidate_batch_type=dto.LunaClassificationCandidateBatch,
        )
        n_logged = 0
        performance_metrics = np.float32(0.0)
        n_processed_training_samples = (epoch - 1) * dataset_length

        for _batch_index, batch in batch_iter:
            self.optimizer.zero_grad()

            loss, batch_metrics = self.compute_batch_loss(batch=batch)

            loss.backward()
            self.optimizer.step()
            if n_processed_training_samples > self.log_every_n_examples * n_logged:
                performance_metrics = self.log_metrics(
                    epoch=epoch,
                    n_processed_training_samples=n_processed_training_samples,
                    mode=enums.Mode.TRAINING,
                    metrics=batch_metrics,
                )
                n_logged += 1
                epoch_metrics.add_batch_metrics(
                    loss=batch_metrics.loss,
                    labels=batch_metrics.labels,
                    predictions=batch_metrics.predictions,
                )
            n_processed_training_samples += len(batch.candidate)

        performance_metrics = self.log_metrics(
            epoch=epoch,
            n_processed_training_samples=n_processed_training_samples,
            mode=enums.Mode.TRAINING,
            metrics=epoch_metrics,
        )
        return performance_metrics

    def do_validation(
        self,
        epoch: int,
        validation_dataloader: data_utils.DataLoader[dto.LunaClassificationCandidate],
    ) -> scoring.PerformanceMetrics:
        with torch.no_grad():
            self.module.eval()
            dataset_length = len(validation_dataloader.dataset)  # type: ignore
            epoch_metrics = scoring.ClassificationMetrics.create_empty(
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
                _, batch_metrics = self.compute_batch_loss(batch=batch)
                if n_processed_training_samples > self.log_every_n_examples * n_logged:
                    score = self.log_metrics(
                        epoch=epoch,
                        n_processed_training_samples=n_processed_training_samples,
                        mode=enums.Mode.VALIDATING,
                        metrics=batch_metrics,
                    )
                    n_logged += 1
                    epoch_metrics.add_batch_metrics(
                        loss=batch_metrics.loss,
                        labels=batch_metrics.labels,
                        predictions=batch_metrics.predictions,
                    )
                n_processed_training_samples += len(batch.candidate)

            score = self.log_metrics(
                epoch=epoch,
                n_processed_training_samples=n_processed_training_samples,
                mode=enums.Mode.VALIDATING,
                metrics=epoch_metrics,
            )
        return score

    def get_module(self) -> torch.nn.Module:
        return self.module

    def compute_batch_loss(
        self, batch: dto.LunaClassificationCandidateBatch
    ) -> tuple[torch.Tensor, scoring.ClassificationMetrics]:
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

        batch_metrics = scoring.ClassificationMetrics(
            loss=loss,
            labels=true_labels,
            predictions=true_probability,
            device=self.device,
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
        metrics: scoring.ClassificationMetrics,
        classification_threshold: float = 0.5,
    ) -> scoring.PerformanceMetrics:
        negative_label_mask = metrics.labels == 0
        negative_prediction_mask = metrics.predictions <= classification_threshold

        positive_label_mask = metrics.labels == 1
        positive_prediction_mask = ~negative_prediction_mask

        epoch_metric: dict[str, dto.NumberMetric] = {}
        epoch_metric.update(
            metrics.prepare_metrics(
                negative_label_mask=negative_label_mask,
                positive_label_mask=positive_label_mask,
            )
        )

        cf = scoring.ConfusionMatrix.create_from_masks(
            negative_label_mask=negative_label_mask,
            positive_label_mask=positive_label_mask,
            positive_prediction_mask=positive_prediction_mask,
            negative_prediction_mask=negative_prediction_mask,
        )
        epoch_metric.update(cf.prepare_metrics())

        performance_scores = scoring.PerformanceMetrics.from_confusion_matrix(cf)
        epoch_metric.update(performance_scores.prepare_metrics())

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

        return performance_scores

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
