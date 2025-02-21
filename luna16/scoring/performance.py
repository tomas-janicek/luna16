import typing

import numpy as np

from luna16 import dto

from . import confusion_matrix


class PerformanceMetrics:
    def __init__(
        self,
        *,
        precision: dto.FloatType,
        recall: dto.FloatType,
        f1_score: dto.FloatType,
        accuracy: dto.FloatType,
        specificity: dto.FloatType,
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.specificity = specificity

    @classmethod
    def from_confusion_matrix(cls, cf: confusion_matrix.ConfusionMatrix) -> typing.Self:
        """Compute classification performance metrics from a confusion matrix.

        This function calculates precision, recall, F1 score, accuracy, and specificity based on the counts provided
        in the confusion matrix object 'cf'. The values are computed while handling potential division by zero errors,
        assigning a value of 0.0 in such cases. Floating point division is performed using np.float32 to ensure proper
        precision.

        Parameters:
            cf (dto.ConfusionMatrix): An object containing the following attributes:
                - tp: Number of true positive predictions.
                - fp: Number of false positive predictions.
                - tn: Number of true negative predictions.
                - fn: Number of false negative predictions.

        Notes:
            - Each metric is computed to handle division by zero, assigning a metric value of 0.0 if its denominator is zero.
            - Floating point division is performed using np.float32 to ensure proper precision.
        """
        denominator_precision = cf.tp + cf.fp
        precision = (
            cf.tp / np.float32(denominator_precision)
            if denominator_precision != 0
            else 0.0
        )
        denominator_recall = cf.tp + cf.fn
        recall = (
            cf.tp / np.float32(denominator_recall) if denominator_recall != 0 else 0.0
        )
        if (precision + recall) != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        total_count = cf.tp + cf.tn + cf.fp + cf.fn
        accuracy = (
            (cf.tp + cf.tn) / np.float32(total_count) if total_count != 0 else 0.0
        )

        denominator_specificity = cf.tn + cf.fp
        specificity = (
            cf.tn / np.float32(denominator_specificity)
            if denominator_specificity != 0
            else 0.0
        )

        return cls(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            specificity=specificity,
        )

    def prepare_metrics(self) -> typing.Mapping[str, dto.NumberMetric]:
        performance_metrics: dict[str, dto.NumberMetric] = {}
        performance_metrics["pr/precision"] = dto.NumberMetric(
            name="Precision",
            value=self.precision,
            formatted_value=f"{self.precision:-5.4f}",
        )
        performance_metrics["pr/recall"] = dto.NumberMetric(
            name="Recall", value=self.recall, formatted_value=f"{self.recall:-5.4f}"
        )
        performance_metrics["pr/f1_score"] = dto.NumberMetric(
            name="F1 Score",
            value=self.f1_score,
            formatted_value=f"{self.f1_score:-5.4f}",
        )
        performance_metrics["pr/accuracy"] = dto.NumberMetric(
            name="Accuracy",
            value=self.accuracy,
            formatted_value=f"{self.accuracy:-5.4f}",
        )
        performance_metrics["pr/specificity"] = dto.NumberMetric(
            name="Specificity",
            value=self.specificity,
            formatted_value=f"{self.specificity:-5.4f}",
        )
        return performance_metrics
