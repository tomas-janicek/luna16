import logging
import typing
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
from torch.utils import data as data_utils

from luna16 import batch_iterators, training_logging

from .. import augmentations, dto, enums, models, utils
from . import base

_log = logging.getLogger(__name__)

_cuda_device = torch.device("cuda")


class SegmentationTrainingAPI(base.BaseTrainingAPI):
    def __init__(
        self,
        *,
        model: models.UNetNormalized,
        optimizer: torch.optim.Optimizer,
        num_workers: int,
        batch_size: int,
        segmentation_logger: training_logging.SegmentationLoggingAdapter,
        batch_iterator: batch_iterators.BaseIteratorProvider,
        training_name: str,
        recall_loss_weight: float = 8,
        augmentation_model: augmentations.SegmentationAugmentation | None = None,
        model_saver: base.ModelSaver | None = None,
    ) -> None:
        self.training_name = training_name
        self.model: models.UNetNormalized | nn.DataParallel[models.UNetNormalized] = (
            model
        )
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.augmentation_model = augmentation_model
        self.n_processed_training_samples = 0
        self.recall_loss_weight = recall_loss_weight
        self.training_started_at = datetime.now()
        self.segmentation_logger = segmentation_logger
        self.batch_iterator = batch_iterator
        self.model_saver = model_saver

        # Initialize model on GPU if available
        self.device = utils.get_device()
        self.n_gpu_device = 1
        if self.is_using_cuda:
            self.n_gpu_device = torch.cuda.device_count()
            _log.info(f"Using CUDA; {self.n_gpu_device} devices.")
            if self.n_gpu_device > 1:
                self.model = nn.DataParallel(module=self.model)
        self.model = model.to(self.device)
        self.modules: typing.Mapping[str, nn.Module] = {}
        if self.augmentation_model:
            self.modules["augmentation_model"] = self.augmentation_model

    @property
    def is_using_cuda(self) -> bool:
        return self.device == _cuda_device

    @classmethod
    def create_with_optimizer_and_model(
        cls,
        num_workers: int,
        batch_size: int,
        augmentation_model: augmentations.SegmentationAugmentation,
        segmentation_logger: training_logging.SegmentationLoggingAdapter,
        training_name: str,
    ) -> "SegmentationTrainingAPI":
        model = models.UNetNormalized(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode=enums.UpMode.UP_CONV,
        )
        # Adam maintains a separate learning rate for each parameter and automatically
        # updates that learning rate as training progresses. Due to these automatic updates,
        # we typically won't need to specify a non-default learning rate when using Adam,
        # since it will quickly determine a reasonable learning rate by itself.
        optimizer = torch.optim.Adam(model.parameters())
        batch_iterator = batch_iterators.BatchIteratorProvider(
            batch_loggers=segmentation_logger.batch_loggers
        )
        return cls(
            model=model,
            optimizer=optimizer,
            num_workers=num_workers,
            batch_size=batch_size,
            augmentation_model=augmentation_model,
            segmentation_logger=segmentation_logger,
            batch_iterator=batch_iterator,
            training_name=training_name,
        )

    def start_training(
        self,
        *,
        epochs: int,
        train: data_utils.Dataset[dto.LunaSegmentationCandidate],
        validation: data_utils.Dataset[dto.LunaSegmentationCandidate],
        validation_cadence: int = 5,
    ) -> None:
        train_dl = self.get_training_dataloader(train=train)
        validation_dl = self.get_validation_dataloader(validation=validation)

        self.segmentation_logger.log_start_training(
            training_api=self,
            n_epochs=epochs,
            batch_size=self.batch_size,
            train_dl=train_dl,
            validation_dl=validation_dl,
        )

        best_score = 0.0
        for epoch in range(1, epochs + 1):
            self.segmentation_logger.log_epoch(epoch=epoch)

            training_metrics = self.do_training(epoch, train_dl)
            self.log_metrics(epoch, enums.Mode.TRAINING, training_metrics)

            if epoch == 1 or epoch % validation_cadence == 0:
                self.do_validation_and_log_results(
                    train_dl=train_dl,
                    validation_dl=validation_dl,
                    best_score=best_score,
                    epoch=epoch,
                )

        self.segmentation_logger.close_all()

    def do_validation_and_log_results(
        self,
        *,
        epoch: int,
        best_score: float,
        train_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        validation_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    ) -> None:
        validation_metrics = self.do_validation(
            epoch=epoch, validation_dataloader=validation_dl
        )
        score = self.log_metrics(epoch, enums.Mode.VALIDATING, validation_metrics)
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

        self.segmentation_logger.log_images(
            epoch=epoch,
            mode=enums.Mode.TRAINING,
            n_processed_samples=self.n_processed_training_samples,
            dataloader=train_dl,
            model=self.model,
            device=self.device,
        )
        self.segmentation_logger.log_images(
            epoch=epoch,
            mode=enums.Mode.VALIDATING,
            n_processed_samples=self.n_processed_training_samples,
            dataloader=validation_dl,
            model=self.model,
            device=self.device,
        )

    def do_training(
        self,
        epoch: int,
        train_dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    ) -> dto.SegmentationBatchMetrics:
        self.model.train()
        train_dataloader.dataset.shuffle_samples()  # type: ignore
        dataset_length = len(train_dataloader.dataset)  # type: ignore
        batch_metrics = dto.SegmentationBatchMetrics.create_empty(
            dataset_len=dataset_length,
            device=self.device,
        )

        batch_iter = self.batch_iterator.enumerate_batches(
            train_dataloader, epoch=epoch, mode=enums.Mode.TRAINING
        )
        for _, batch in batch_iter:
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
        validation_dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    ) -> dto.SegmentationBatchMetrics:
        with torch.no_grad():
            self.model.eval()
            dataset_length = len(validation_dataloader.dataset)  # type: ignore
            batch_metrics = dto.SegmentationBatchMetrics.create_empty(
                dataset_len=dataset_length,
                device=self.device,
            )

            batch_iter = self.batch_iterator.enumerate_batches(
                validation_dataloader, epoch=epoch, mode=enums.Mode.VALIDATING
            )
            for _, batch in batch_iter:
                _, batch_metrics = self.compute_batch_loss(
                    batch=batch,
                    batch_metrics=batch_metrics,
                )

        return batch_metrics

    def compute_batch_loss(
        self,
        batch: dto.LunaSegmentationCandidate,
        batch_metrics: dto.SegmentationBatchMetrics,
        classification_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, dto.SegmentationBatchMetrics]:
        input: torch.Tensor = batch.candidate.to(self.device, non_blocking=True)
        label: torch.Tensor = batch.positive_candidate_mask.to(
            self.device, non_blocking=True
        )

        if self.model.training and self.augmentation_model:
            input, label = self.augmentation_model(input, label)

        prediction: torch.Tensor = self.model(input)

        dice_loss = self.get_dice_loss(prediction, label)
        # The dice loss below represent only loss for false negative pixels. Everything
        # that should have been predicted `True` but wasn't.
        false_negative_loss = self.get_dice_loss(prediction * label, label)

        with torch.no_grad():
            # We threshold the prediction to get “hard” Dice but convert
            # to float for the later multiplication. We are checking second
            # dimension of `prediction` representing chanel and creating new tensor
            # with `True` everywhere this chanel has values greater than `classification_threshold`.
            prediction_hard = (prediction[:, 0:1] > classification_threshold).to(
                torch.float32
            )

            hard_true_positive = (prediction_hard * label).sum(dim=[1, 2, 3])
            hard_false_negative = ((1 - prediction_hard) * label).sum(dim=[1, 2, 3])
            hard_false_positive = (prediction_hard * (~label)).sum(dim=[1, 2, 3])

            batch_metrics.add_batch_metrics(
                loss=dice_loss,
                false_negative_loss=false_negative_loss,
                hard_true_positive=hard_true_positive,
                hard_false_negative=hard_false_negative,
                hard_false_positive=hard_false_positive,
            )

        # Since the area covered by the positive mask is much, much smaller than
        # the whole cropped image, we want to "boost" `false_negative_loss` loss
        # by multiplying it with `recall_loss_weight` constant.
        # WARNING: This can only by done when using Adam optimizer.
        return (
            dice_loss.mean() - false_negative_loss.mean() * self.recall_loss_weight,
            batch_metrics,
        )

    def get_dice_loss(
        self, prediction: torch.Tensor, label: torch.Tensor, epsilon: int = 1
    ) -> torch.Tensor:
        """Function calculates dice loss using equation:
        Dice = (2 * TP) / (2 * TP + FP + FN)
        In our case, TP is equal to `correct`,
        `prediction_aggregated` is equal to TP + FP,
        and label_aggregated is equal to TP + FN.
        Thats why Dice = (2 * correct) / (prediction_aggregated + label_aggregated)"""

        # Sums over everything except the batch dimension to get
        # the positively labeled, (softly) positively detected, and
        # (softly) correct positives per batch item
        label_aggregated = label.sum(dim=[1, 2, 3])
        prediction_aggregated = prediction.sum(dim=[1, 2, 3])
        correct = (prediction * label).sum(dim=[1, 2, 3])

        # We add epsilon to both nominator and denominator to avoid dividing with zero
        dice_ratio = (2 * correct + epsilon) / (
            prediction_aggregated + label_aggregated + epsilon
        )
        # To make it a loss,we take `1 - Dice` ratio, so 0 represent the best outcome.
        return 1 - dice_ratio

    def get_training_dataloader(
        self, train: data_utils.Dataset[dto.LunaSegmentationCandidate]
    ) -> data_utils.DataLoader[dto.LunaSegmentationCandidate]:
        batch_size = self.batch_size * self.n_gpu_device
        train_dataloader = data_utils.DataLoader(
            dataset=train,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.is_using_cuda,
        )
        return train_dataloader

    def get_validation_dataloader(
        self, validation: data_utils.Dataset[dto.LunaSegmentationCandidate]
    ) -> data_utils.DataLoader[dto.LunaSegmentationCandidate]:
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
        metrics: dto.SegmentationBatchMetrics,
    ) -> float:
        epoch_metric: dict[str, training_logging.NumberValue] = {}

        n_true_positives = metrics.true_positive.sum(0).item()
        n_false_negative = metrics.false_negative.sum(0).item()
        n_false_positive = metrics.false_positive.sum(0).item()
        n_all_labels = (n_true_positives + n_false_negative) or 1.0

        loss = metrics.loss.mean().item()
        epoch_metric["loss/all"] = training_logging.NumberValue(
            name="Loss", value=loss, formatted_value=f"{loss:-5.4f}"
        )
        true_positives = n_true_positives / n_all_labels
        epoch_metric["percent_all/tp"] = training_logging.NumberValue(
            name="True Positive",
            value=true_positives,
            formatted_value=f"{true_positives:.0%}",
        )
        false_negatives = n_false_negative / n_all_labels
        epoch_metric["percent_all/fn"] = training_logging.NumberValue(
            name="False Negative",
            value=false_negatives,
            formatted_value=f"{false_negatives:.0%}",
        )

        false_positive = n_false_positive / n_all_labels
        epoch_metric["percent_all/fp"] = training_logging.NumberValue(
            name="False Positive",
            value=false_positive,
            formatted_value=f"{false_positive:.0%}",
        )

        precision = n_true_positives / ((n_true_positives + n_false_positive) or 1)
        recall = n_true_positives / ((n_true_positives + n_false_negative) or 1)
        epoch_metric["pr/precision"] = training_logging.NumberValue(
            name="Precision", value=precision, formatted_value=f"{precision:-5.4f}"
        )
        epoch_metric["pr/recall"] = training_logging.NumberValue(
            name="Recall", value=recall, formatted_value=f"{recall:-5.4f}"
        )
        f1_score = 2 * (precision * recall) / ((precision + recall) or 1)
        epoch_metric["pr/f1_score"] = training_logging.NumberValue(
            name="F1 Score", value=f1_score, formatted_value=f"{f1_score:-5.4f}"
        )

        self.segmentation_logger.log_metrics(
            epoch=epoch,
            mode=mode,
            n_processed_samples=self.n_processed_training_samples,
            values=epoch_metric,
        )

        return recall

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"augmentation_model={self.augmentation_model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"device={self.device}, "
            f"n_gpu_device={self.n_gpu_device}, "
            f"num_workers={self.num_workers}, "
            f"batch_size={self.batch_size})"
        )
