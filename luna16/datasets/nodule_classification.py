import copy
import logging
import random

import torch
import torch.nn.functional as F
from torch.utils import data as data_utils

from luna16 import augmentations, dto

from . import utils

_log = logging.getLogger(__name__)


def _candidate_key(candidate: dto.CandidateInfo) -> str:
    return candidate.series_uid


class LunaDataset(data_utils.Dataset[dto.LunaClassificationCandidate]):
    def __init__(
        self,
        train: bool = True,
        validation_stride: int = 20,
        ratio: int = 2,
        series_uids: list[str] | None = None,
        training_length: int | None = None,
        transformations: list[augmentations.Transformation] | None = None,
        filters: list[augmentations.Filter] | None = None,
    ) -> None:
        self.candidates_info: list[dto.CandidateInfo] = copy.copy(
            utils.get_candidates_info()
        )
        self.train = train
        self.validation_stride = validation_stride
        self.ratio = ratio
        self.training_length = training_length
        self.transformations = transformations
        self.filters = filters

        if validation_stride <= 0 or validation_stride > len(self.candidates_info):
            raise ValueError(
                "Argument validation_stride must have value greater than 0 and less than "
                f"{len(self.candidates_info)} (length of the dataset)"
            )

        # 400 is in this case arbitrary number
        if ratio <= 0 or ratio >= 400:
            raise ValueError(
                "Argument ratio must have value between 0 and 400 for "
                "optimal performance."
            )

        # Filter candidates by dataset arguments (for example select CT with specific Series UIDs)
        if series_uids:
            self.candidates_info = [
                x for x in self.candidates_info if x.series_uid in series_uids
            ]

        if train:
            del self.candidates_info[::validation_stride]
        else:
            self.candidates_info = self.candidates_info[::validation_stride]

        # Split candidates into is nodule and is not nodule lists so we can control how many
        # positive candidates we use. This is necessary because our dataset does not have
        # enough positive samples for training (ration 400:1).
        self.is_nodule_candidates = [
            candidate for candidate in self.candidates_info if candidate.is_nodule
        ]
        self.not_nodule_candidates = [
            candidate for candidate in self.candidates_info if not candidate.is_nodule
        ]

        if not self.is_nodule_candidates or not self.not_nodule_candidates:
            raise ValueError(
                f"LuNA dataset must have at least 1 positive and 1 negative sample "
                f"and it has {len(self.is_nodule_candidates)} positive and "
                f"{len(self.not_nodule_candidates)} negative candidates."
            )

        _log.info("%s", repr(self))

    def shuffle_candidates(self) -> None:
        random.shuffle(self.is_nodule_candidates)
        random.shuffle(self.not_nodule_candidates)

    def sort_by_series_uid(self) -> None:
        self.is_nodule_candidates.sort(key=_candidate_key)
        self.not_nodule_candidates.sort(key=_candidate_key)

    def __len__(self) -> int:
        return (
            self.training_length if self.training_length else len(self.candidates_info)
        )

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        cutout_shape = dto.CoordinatesIRC(index=32, row=48, col=48)
        candidate_array, _positive_candidate_array, center_irc = (
            utils.Ct.create_ct_and_get_raw_image(
                series_uid=candidate_info.series_uid,
                center=candidate_info.center,
                cutout_shape=cutout_shape,
            )
        )
        candidate = torch.from_numpy(candidate_array).to(torch.float32).unsqueeze(0)

        candidate = self._apply_all_transformations(candidate)
        candidate = self._apply_all_filters(candidate)

        actual_result = torch.tensor(
            [not candidate_info.is_nodule, candidate_info.is_nodule],
            dtype=torch.long,
        )

        return dto.LunaClassificationCandidate(
            candidate=candidate,
            labels=actual_result,
            series_uid=candidate_info.series_uid,
            center_irc=torch.tensor(center_irc),
        )

    def _apply_all_filters(self, candidate: torch.Tensor) -> torch.Tensor:
        if self.filters:
            for filter in self.filters:
                candidate = filter.apply_filter(image=candidate)
        return candidate

    def _apply_all_transformations(self, candidate: torch.Tensor) -> torch.Tensor:
        if self.transformations:
            transformation_matrix = torch.eye(4)
            for transformation in self.transformations:
                transformation_matrix = transformation.apply_transformation(
                    transformation=transformation_matrix
                )

            batched_transformation_matrix = transformation_matrix[:3].unsqueeze(dim=0)
            batched_candidate = candidate.unsqueeze(dim=0)
            affine_transformation_tensor = F.affine_grid(
                theta=batched_transformation_matrix,
                size=list(batched_candidate.size()),
                align_corners=False,
            )

            batched_candidate = F.grid_sample(
                input=batched_candidate,
                grid=affine_transformation_tensor,
                padding_mode="border",
                align_corners=False,
            )

            # Un-batching the only candidate returned from grid_sample
            return batched_candidate[0]
        return candidate

    def _get_candidate_info(self, index: int) -> dto.CandidateInfo:
        is_nodule_index = index // (self.ratio + 1)
        # Negative Index
        if index % (self.ratio + 1):
            not_nodule_index = index - 1 - is_nodule_index
            not_nodule_index: int = not_nodule_index % len(self.not_nodule_candidates)
            candidate_info = self.not_nodule_candidates[not_nodule_index]
        # Positive index
        else:
            is_nodule_index = is_nodule_index % len(self.is_nodule_candidates)
            candidate_info = self.is_nodule_candidates[is_nodule_index]

        return candidate_info

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"len={len(self.candidates_info)}, "
            f"training_len={self.training_length}, "
            f"positive_len={len(self.is_nodule_candidates)}, "
            f"negative_len={len(self.not_nodule_candidates)}, "
            f"train={self.train}, "
            f"validation_stride={self.validation_stride})"
        )
        return _repr
