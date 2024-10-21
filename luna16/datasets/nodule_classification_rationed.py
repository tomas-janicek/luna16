import copy
import datetime
import logging
import random

import torch
import torch.nn.functional as F
from profilehooks import profile
from torch.utils import data as data_utils

from luna16 import augmentations, dto, enums, settings
from luna16 import utils as common_utils

from . import utils

_log = logging.getLogger(__name__)


def _candidate_key(candidate: dto.CandidateMalignancyInfo) -> str:
    return candidate.series_uid


class LunaRationedDataset(data_utils.Dataset[dto.LunaClassificationCandidate]):
    def __init__(
        self,
        ratio: dto.Ratio,
        train: bool = True,
        validation_stride: int = 20,
        training_length: int | None = None,
        transformations: list[augmentations.Transformation] | None = None,
        filters: list[augmentations.Filter] | None = None,
    ) -> None:
        self.train = train
        self.validation_stride = validation_stride
        self.ratio = ratio
        self.training_length = training_length
        self.transformations = transformations
        self.filters = filters

        candidates_info: list[dto.CandidateMalignancyInfo] = copy.copy(
            utils.get_candidates_with_malignancy_info()
        )
        self.series_uids: list[str] = sorted(
            utils.get_grouped_candidates_with_malignancy_info(
                length=self.training_length
            ).keys()
        )
        if validation_stride <= 0 or validation_stride > len(candidates_info):
            raise ValueError(
                "Argument validation_stride must have value greater than 0 and less than "
                f"{len(candidates_info)} (length of the dataset)"
            )

        # 400 is in this case arbitrary number
        if ratio.cycle <= 0 or ratio.cycle >= 400:
            raise ValueError(
                "Argument ratio must have value between 0 and 400 for "
                "optimal performance."
            )

        if train:
            del self.series_uids[::validation_stride]
        else:
            self.series_uids = self.series_uids[::validation_stride]

        # Update candidates_info so it reflects changes done to series_uids.
        series_uids_set = set(self.series_uids)
        self.candidates_info = [
            candidate
            for candidate in candidates_info
            if candidate.series_uid in series_uids_set
        ]

        # Split candidates into is nodule and is not nodule lists so we can control how many
        # positive candidates we use. This is necessary because our dataset does not have
        # enough positive samples for training (ration 400:1).
        self.is_nodule_candidates = [
            candidate for candidate in self.candidates_info if candidate.is_nodule
        ]
        self.not_nodule_candidates = [
            candidate for candidate in self.candidates_info if not candidate.is_nodule
        ]
        self.is_malignant_candidates = [
            candidate for candidate in self.candidates_info if candidate.is_malignant
        ]
        self.not_malignant_candidates = [
            candidate
            for candidate in self.candidates_info
            if not candidate.is_malignant
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
        random.shuffle(self.is_malignant_candidates)
        random.shuffle(self.not_malignant_candidates)

    def sort_by_series_uid(self) -> None:
        self.is_nodule_candidates.sort(key=_candidate_key)
        self.not_nodule_candidates.sort(key=_candidate_key)
        self.is_malignant_candidates.sort(key=_candidate_key)
        self.not_malignant_candidates.sort(key=_candidate_key)

    def __len__(self) -> int:
        return (
            self.training_length if self.training_length else len(self.candidates_info)
        )

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        return self._create_luna_candidate(candidate_info)

    @profile(
        filename=settings.PROFILING_DIR
        / f"dataloader_profile_{common_utils.get_datetime_string(datetime.datetime.now())}.prof",
        stdout=False,
        dirs=True,
    )  # type: ignore
    def _create_luna_candidate(
        self, candidate_info: dto.CandidateMalignancyInfo
    ) -> dto.LunaClassificationCandidate:
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

    def _get_candidate_info(self, index: int) -> dto.CandidateMalignancyInfo:
        candidate_type, typed_index = self.ratio.get_class(index)
        match enums.LunaCandidateTypes(candidate_type):
            case enums.LunaCandidateTypes.POSITIVE:
                is_nodule_index: int = typed_index % len(self.is_nodule_candidates)
                return self.is_nodule_candidates[is_nodule_index]
            case enums.LunaCandidateTypes.NEGATIVE:
                not_nodule_index: int = typed_index % len(self.not_nodule_candidates)
                return self.not_nodule_candidates[not_nodule_index]

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


class MalignantLunaDataset(LunaRationedDataset):
    def __len__(self):
        return (
            self.training_length
            if self.training_length
            else len(self.not_malignant_candidates + self.is_malignant_candidates)
        )

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        return self._create_luna_candidate(candidate_info)

    def _get_candidate_info(self, index: int) -> dto.CandidateMalignancyInfo:
        candidate_type, typed_index = self.ratio.get_class(index)
        match enums.LunaMalignantCandidateTypes(candidate_type):
            case enums.LunaMalignantCandidateTypes.MALIGNANT:
                is_malignant_index: int = typed_index % len(
                    self.is_malignant_candidates
                )
                return self.is_malignant_candidates[is_malignant_index]
            case enums.LunaMalignantCandidateTypes.BENIGN:
                not_malignant_index: int = typed_index % len(
                    self.not_malignant_candidates
                )
                return self.not_malignant_candidates[not_malignant_index]
            case enums.LunaMalignantCandidateTypes.NOT_NODULE:
                not_nodule_index: int = typed_index % len(self.not_nodule_candidates)
                return self.not_nodule_candidates[not_nodule_index]
