import torch
import torch.nn as nn
from mlflow.models import infer_signature
from mlflow.pytorch import ModelSignature
from torch.utils import data as data_utils

from luna16 import augmentations, training_logging, utils
from luna16.batch_iterators.batch_iterator import BatchIteratorProvider

from .. import dto, enums
from . import base


class NoduleSegmentationModel(base.BaseModel[dto.LunaSegmentationCandidate]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.optimizer.Optimizer,
        batch_iterator: BatchIteratorProvider,
        logger: training_logging.LogMessageHandler,
        validation_cadence: int = 5,
        recall_loss_weight: float = 8,
        augmentation_model: augmentations.SegmentationAugmentation | None = None,
    ) -> None:
        self.device, n_gpu_devices = utils.get_device()
        self.model = model
        if n_gpu_devices > 1:
            self.model = nn.DataParallel(module=self.model)
        self.model = self.model.to(self.device)
        self.optimizer = optimizer
        self.validation_cadence = validation_cadence
        self.batch_iterator = batch_iterator
        self.recall_loss_weight = recall_loss_weight
        self.logger = logger
        self.augmentation_model = augmentation_model

    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        validation_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    ) -> dto.Scores:
        dataset_length = len(train_dl.dataset)  # type: ignore
        training_metrics = self.do_training(epoch, train_dl)
        score = self.log_metrics(
            epoch=epoch,
            n_processed_training_samples=epoch * dataset_length,
            mode=enums.Mode.TRAINING,
            metrics=training_metrics,
        )

        if epoch == 1 or epoch % self.validation_cadence == 0:
            self.do_validation_and_log_results(
                train_dl=train_dl,
                validation_dl=validation_dl,
                epoch=epoch,
            )
        return {"score": score}

    def do_validation_and_log_results(
        self,
        *,
        epoch: int,
        train_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        validation_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    ) -> None:
        dataset_length = len(train_dl.dataset)  # type: ignore
        validation_metrics = self.do_validation(
            epoch=epoch, validation_dataloader=validation_dl
        )
        n_processed_training_samples = epoch * dataset_length
        _score = self.log_metrics(
            epoch=epoch,
            n_processed_training_samples=n_processed_training_samples,
            mode=enums.Mode.VALIDATING,
            metrics=validation_metrics,
        )

        val_log_images = training_logging.LogImages(
            epoch=epoch,
            mode=enums.Mode.TRAINING,
            n_processed_samples=n_processed_training_samples,
            dataloader=train_dl,
            model=self.model,
            device=self.device,
        )
        self.logger.handle_message(val_log_images)
        train_log_images = training_logging.LogImages(
            epoch=epoch,
            mode=enums.Mode.VALIDATING,
            n_processed_samples=n_processed_training_samples,
            dataloader=validation_dl,
            model=self.model,
            device=self.device,
        )
        self.logger.handle_message(train_log_images)

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

    def get_module(self) -> nn.Module:
        return self.model

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

    def get_signature(
        self, train_dl: data_utils.DataLoader[dto.LunaSegmentationCandidate]
    ) -> ModelSignature:
        input = torch.unsqueeze(train_dl.dataset[0].candidate, 0)
        input = input.to(self.device, non_blocking=True)
        _logits, probability = self.model(input)
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
        metrics: dto.SegmentationBatchMetrics,
    ) -> float:
        epoch_metric: dict[str, dto.NumberValue] = {}

        n_true_positives = metrics.true_positive.sum(0).item()
        n_false_negative = metrics.false_negative.sum(0).item()
        n_false_positive = metrics.false_positive.sum(0).item()
        n_all_labels = (n_true_positives + n_false_negative) or 1.0

        loss = metrics.loss.mean().item()
        epoch_metric["loss/all"] = dto.NumberValue(
            name="Loss", value=loss, formatted_value=f"{loss:-5.4f}"
        )
        true_positives = n_true_positives / n_all_labels
        epoch_metric["percent_all/tp"] = dto.NumberValue(
            name="True Positive",
            value=true_positives,
            formatted_value=f"{true_positives:.0%}",
        )
        false_negatives = n_false_negative / n_all_labels
        epoch_metric["percent_all/fn"] = dto.NumberValue(
            name="False Negative",
            value=false_negatives,
            formatted_value=f"{false_negatives:.0%}",
        )

        false_positive = n_false_positive / n_all_labels
        epoch_metric["percent_all/fp"] = dto.NumberValue(
            name="False Positive",
            value=false_positive,
            formatted_value=f"{false_positive:.0%}",
        )

        precision = n_true_positives / ((n_true_positives + n_false_positive) or 1)
        recall = n_true_positives / ((n_true_positives + n_false_negative) or 1)
        epoch_metric["pr/precision"] = dto.NumberValue(
            name="Precision", value=precision, formatted_value=f"{precision:-5.4f}"
        )
        epoch_metric["pr/recall"] = dto.NumberValue(
            name="Recall", value=recall, formatted_value=f"{recall:-5.4f}"
        )
        f1_score = 2 * (precision * recall) / ((precision + recall) or 1)
        epoch_metric["pr/f1_score"] = dto.NumberValue(
            name="F1 Score", value=f1_score, formatted_value=f"{f1_score:-5.4f}"
        )

        log_metrics = training_logging.LogMetrics(
            epoch=epoch,
            mode=mode,
            n_processed_samples=n_processed_training_samples,
            values=epoch_metric,
        )
        self.logger.handle_message(log_metrics)

        return recall
