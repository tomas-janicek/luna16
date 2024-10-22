import logging

import numpy as np
import torch

from luna16 import dto, enums

from .nodule_cutouts import CutoutsDataset

_log = logging.getLogger(__name__)


class MalignantCutoutsDataset(CutoutsDataset):
    def __len__(self):
        return len(self.not_malignant_candidates + self.is_malignant_candidates)

    def __getitem__(self, index: int) -> dto.LunaClassificationCandidate:
        candidate_info = self._get_candidate_info(index)

        return self._create_luna_candidate(candidate_info)

    def _get_candidate_info(self, index: int) -> dto.CandidateMetadata:
        candidate_type, typed_index = self.ratio.get_class(index)
        match enums.CandidateClass(candidate_type):
            case enums.CandidateClass.MALIGNANT:
                is_malignant_index: int = typed_index % len(
                    self.is_malignant_candidates
                )
                nodule_metadata = self.is_malignant_candidates[is_malignant_index]
            case enums.CandidateClass.BENIGN:
                not_malignant_index: int = typed_index % len(
                    self.not_malignant_candidates
                )
                nodule_metadata = self.not_malignant_candidates[not_malignant_index]
            case enums.CandidateClass.NOT_NODULE:
                not_nodule_index: int = typed_index % len(self.not_nodule_candidates)
                nodule_metadata = self.not_nodule_candidates[not_nodule_index]

        return dto.CandidateMetadata(
            series_uid=nodule_metadata["seriesuid"],
            candidate_class=nodule_metadata["class"],
            diameter_mm=nodule_metadata["diameter_mm"],
            file_path=nodule_metadata["file_path"],
        )

    def _create_luna_candidate(
        self, candidate_metadata: dto.CandidateMetadata
    ) -> dto.LunaClassificationCandidate:
        loaded_cutout = np.load(candidate_metadata.file_path)
        candidate_data = (
            torch.from_numpy(loaded_cutout["ct_chunk"]).to(torch.float32).unsqueeze(0)
        )

        match candidate_metadata.candidate_class:
            case enums.CandidateClass.MALIGNANT:
                actual_result = torch.tensor([0, 1], dtype=torch.long)
            case enums.CandidateClass.BENIGN | enums.CandidateClass.NOT_NODULE:
                actual_result = torch.tensor([1, 0], dtype=torch.long)

        candidate_data = self._apply_all_transformations(candidate_data)
        candidate_data = self._apply_all_filters(candidate_data)

        candidate = dto.LunaClassificationCandidate(
            series_uid=candidate_metadata.series_uid,
            candidate=candidate_data,
            labels=actual_result,
            center_irc=torch.tensor(loaded_cutout["center_irc"]),
        )
        return candidate
