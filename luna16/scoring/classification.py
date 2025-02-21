import typing

import torch

from luna16 import dto


class ClassificationMetrics:
    def __init__(
        self,
        loss: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.loss = loss
        self.labels = labels
        self.predictions = predictions
        self.device = device

    @classmethod
    def create_empty(
        cls, dataset_len: int, device: torch.device
    ) -> "ClassificationMetrics":
        return cls(
            loss=torch.tensor([]).to(device),
            labels=torch.tensor([]).to(device),
            predictions=torch.tensor([]).to(device),
            device=device,
        )

    def add_batch_metrics(
        self,
        loss: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        predictions: torch.Tensor | None = None,
    ) -> None:
        if loss is not None:
            self.loss = torch.cat((self.loss, loss), dim=0)
        if labels is not None:
            self.labels = torch.cat((self.labels, labels), dim=0)
        if predictions is not None:
            self.predictions = torch.cat((self.predictions, predictions), dim=0)

    def dataset_length(self) -> int:
        return self.labels.shape[0]

    def prepare_metrics(
        self, negative_label_mask: torch.Tensor, positive_label_mask: torch.Tensor
    ) -> typing.Mapping[str, dto.NumberMetric]:
        loss_metrics: dict[str, dto.NumberMetric] = {}
        loss = self.loss.mean().item()
        loss_metrics["loss/all"] = dto.NumberMetric(
            name="Loss", value=loss, formatted_value=f"{loss:-5.4f}"
        )

        loss_negative = self.loss[negative_label_mask].mean().item()
        loss_metrics["loss/neg"] = dto.NumberMetric(
            name="Loss Negative",
            value=loss_negative,
            formatted_value=f"{loss_negative:-5.4f}",
        )

        loss_positive = self.loss[positive_label_mask].mean().item()
        loss_metrics["loss/pos"] = dto.NumberMetric(
            name="Loss Positive",
            value=loss_positive,
            formatted_value=f"{loss_positive:-5.4f}",
        )
        return loss_metrics
