import typing

import torch

from luna16 import dto


class ConfusionMatrix:
    def __init__(self, *, tp: int, fp: int, tn: int, fn: int) -> None:
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        self.n_negatives = tn + fp
        self.n_positives = tp + fn

        self.n_all = self.n_negatives + self.n_positives

        self.correct_all = tp + tn

        self.correct_all_percent = self.correct_all / self.n_all
        self.correct_positives_percent = tp / self.n_positives
        self.correct_negatives_percent = tn / self.n_negatives

    @classmethod
    def create_from_masks(
        cls,
        *,
        negative_label_mask: torch.Tensor,
        positive_label_mask: torch.Tensor,
        positive_prediction_mask: torch.Tensor,
        negative_prediction_mask: torch.Tensor,
    ) -> "ConfusionMatrix":
        tp = int((positive_label_mask & positive_prediction_mask).sum())
        fp = int((negative_label_mask & positive_prediction_mask).sum())
        tn = int((negative_label_mask & negative_prediction_mask).sum())
        fn = int((positive_label_mask & negative_prediction_mask).sum())
        return cls(tp=tp, fp=fp, tn=tn, fn=fn)

    def prepare_metrics(self) -> typing.Mapping[str, dto.NumberMetric]:
        metrics: dict[str, dto.NumberMetric] = {}
        metrics["correct/all"] = dto.NumberMetric(
            name="Correct All",
            value=self.correct_all_percent,
            formatted_value=f"{self.correct_all_percent:.0%}",
        )
        metrics["correct/neg"] = dto.NumberMetric(
            name="Correct Negative",
            value=self.correct_negatives_percent,
            formatted_value=f"{self.correct_negatives_percent:.0%}",
        )
        metrics["correct/pos"] = dto.NumberMetric(
            name="Correct Positive",
            value=self.correct_positives_percent,
            formatted_value=f"{self.correct_positives_percent:.0%}",
        )
        return metrics
