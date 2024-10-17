import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data as data_utils

from luna16 import augmentations, dto, enums
from luna16.settings import settings

_log = logging.getLogger(__name__)


def get_present_candidates() -> pd.DataFrame:
    present_candidates_path = settings.CACHE_DIR / "luna16" / "present_candidates.csv"
    return pd.read_csv(filepath_or_buffer=present_candidates_path)


class LunaCutoutsDataset(data_utils.Dataset[dto.LunaClassificationCandidate]):
    def __init__(
        self,
        ratio: dto.Ratio,
        train: bool = True,
        validation_stride: int = 20,
        transformations: list[augmentations.Transformation] | None = None,
        filters: list[augmentations.Filter] | None = None,
    ) -> None:
        self.train = train
        self.validation_stride = validation_stride
        self.ratio = ratio
        self.transformations = transformations
        self.filters = filters

        self.candidates_info = get_present_candidates()

        if validation_stride <= 0 or validation_stride > len(self.candidates_info):
            raise ValueError(
                "Argument validation_stride must have value greater than 0 and less than "
                f"{len(self.candidates_info)} (length of the dataset)"
            )

        # 400 is in this case arbitrary number
        if ratio.cycle <= 0 or ratio.cycle >= 400:
            raise ValueError(
                "Argument ratio must have value between 0 and 400 for "
                "optimal performance."
            )

        if train:
            self.candidates_info = self.candidates_info.drop(
                self.candidates_info.index[::validation_stride]
            )
        else:
            self.candidates_info = self.candidates_info.iloc[::validation_stride]

        # Split candidates into is nodule and is not nodule lists so we can control how many
        # positive candidates we use. This is necessary because our dataset does not have
        # enough positive samples for training (ration 400:1).

        self.is_nodule_candidates = self.candidates_info[
            self.candidates_info["is_nodule"] == True  # noqa: E712
        ]
        self.not_nodule_candidates = self.candidates_info[
            self.candidates_info["is_nodule"] == False  # noqa: E712
        ]
        self.is_malignant_candidates = self.candidates_info[
            self.candidates_info["is_malignant"] == True  # noqa: E712
        ]
        self.not_malignant_candidates = self.candidates_info[
            self.candidates_info["is_malignant"] == False  # noqa: E712
        ]

        if self.is_nodule_candidates.empty or self.not_nodule_candidates.empty:
            raise ValueError(
                f"LuNA dataset must have at least 1 positive and 1 negative sample "
                f"and it has {len(self.is_nodule_candidates)} positive and "
                f"{len(self.not_nodule_candidates)} negative candidates."
            )

        _log.info("%s", repr(self))

    def __len__(self) -> int:
        return len(self.candidates_info)

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        return self._create_luna_candidate(candidate_info)

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"len={len(self.candidates_info)}, "
            f"positive_len={len(self.is_nodule_candidates)}, "
            f"negative_len={len(self.not_nodule_candidates)}, "
            f"train={self.train}, "
            f"validation_stride={self.validation_stride})"
        )
        return _repr

    def shuffle_candidates(self) -> None:
        self.is_nodule_candidates.sample(frac=1).reset_index(drop=True)
        self.not_nodule_candidates.sample(frac=1).reset_index(drop=True)
        self.is_malignant_candidates.sample(frac=1).reset_index(drop=True)
        self.not_malignant_candidates.sample(frac=1).reset_index(drop=True)

    def _create_luna_candidate(
        self, candidate_metadata: dto.CandidateMetadata
    ) -> dto.LunaClassificationCandidate:
        loaded_cutout = np.load(candidate_metadata.file_path)
        candidate_data = (
            torch.from_numpy(loaded_cutout["ct_chunk"]).to(torch.float32).unsqueeze(0)
        )
        actual_result = torch.tensor(
            [not candidate_metadata.is_nodule, candidate_metadata.is_nodule],
            dtype=torch.long,
        )

        candidate_data = self._apply_all_transformations(candidate_data)
        candidate_data = self._apply_all_filters(candidate_data)

        candidate = dto.LunaClassificationCandidate(
            series_uid=candidate_metadata.series_uid,
            candidate=candidate_data,
            labels=actual_result,
            center_irc=torch.tensor(loaded_cutout["center_irc"]),
        )
        return candidate

    def _get_candidate_info(self, index: int) -> dto.CandidateMetadata:
        candidate_type, typed_index = self.ratio.get_class(index)
        match enums.LunaCandidateTypes(candidate_type):
            case enums.LunaCandidateTypes.POSITIVE:
                is_nodule_index: int = typed_index % len(self.is_nodule_candidates)
                nodule_metadata = self.is_nodule_candidates.iloc[is_nodule_index]
            case enums.LunaCandidateTypes.NEGATIVE:
                not_nodule_index: int = typed_index % len(self.not_nodule_candidates)
                nodule_metadata = self.not_nodule_candidates.iloc[not_nodule_index]
        return dto.CandidateMetadata(
            series_uid=nodule_metadata["seriesuid"],
            is_nodule=nodule_metadata["is_nodule"],
            is_annotated=nodule_metadata["is_annotated"],
            is_malignant=nodule_metadata["is_malignant"],
            diameter_mm=nodule_metadata["diameter_mm"],
            file_path=nodule_metadata["file_path"],
        )

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

    def _apply_all_filters(self, candidate: torch.Tensor) -> torch.Tensor:
        if self.filters:
            for filter in self.filters:
                candidate = filter.apply_filter(image=candidate)
        return candidate


class MalignantLunaDataset(LunaCutoutsDataset):
    def __len__(self):
        return len(self.not_malignant_candidates + self.is_malignant_candidates)

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        return self._create_luna_candidate(candidate_info)

    def _get_candidate_info(self, index: int) -> dto.CandidateMetadata:
        candidate_type, typed_index = self.ratio.get_class(index)
        match enums.LunaMalignantCandidateTypes(candidate_type):
            case enums.LunaMalignantCandidateTypes.MALIGNANT:
                is_malignant_index: int = typed_index % len(
                    self.is_malignant_candidates
                )
                nodule_metadata = self.is_malignant_candidates[is_malignant_index]
            case enums.LunaMalignantCandidateTypes.BENIGN:
                not_malignant_index: int = typed_index % len(
                    self.not_malignant_candidates
                )
                nodule_metadata = self.not_malignant_candidates[not_malignant_index]
            case enums.LunaMalignantCandidateTypes.NOT_NODULE:
                not_nodule_index: int = typed_index % len(self.not_nodule_candidates)
                nodule_metadata = self.not_nodule_candidates[not_nodule_index]
        return dto.CandidateMetadata(
            series_uid=nodule_metadata["seriesuid"],
            is_nodule=nodule_metadata["is_nodule"],
            is_annotated=nodule_metadata["is_annotated"],
            is_malignant=nodule_metadata["is_malignant"],
            diameter_mm=nodule_metadata["diameter_mm"],
            file_path=nodule_metadata["file_path"],
        )
