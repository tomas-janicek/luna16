import logging
import random

import numpy as np
import torch
from torch.utils import data as data_utils

from luna16 import dto

from . import utils

_log = logging.getLogger(__name__)


class LunaSegmentationDataset(data_utils.Dataset[dto.LunaSegmentationCandidate]):
    def __init__(
        self,
        train: bool = True,
        validation_stride: int = 0,
        n_context_slices: int = 3,
        use_full_ct: bool = False,
        training_length: int | None = None,
    ) -> None:
        self.n_context_slices = n_context_slices
        self.use_full_ct = use_full_ct
        self.train = train
        self.validation_stride = validation_stride
        self.training_length = training_length
        self.series_uids: list[str] = sorted(
            utils.get_grouped_candidates_with_malignancy_info(
                length=self.training_length
            ).keys()
        )

        candidates_info: list[dto.CandidateMalignancyInfo] = (
            utils.get_shortened_candidates_with_malignancy_info(
                length=self.training_length
            )
        )

        if validation_stride <= 0 or validation_stride > len(candidates_info):
            raise ValueError(
                "Argument validation_stride must have value greater than 0 and less than "
                f"{len(candidates_info)} (length of the dataset)"
            )

        # We are deleting whole CT scans with all it's candidates because we are not training on
        # individual candidates but on whole CT scans (2D subset of slices).
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

        # Parameter use_full_ct decides whether we take all slices for CT scan or just take
        # slices that contain any voxel marked as nodule.
        self.series_uid__slice_index: list[tuple[str, int]] = []
        for series_uid in self.series_uids:
            n_ct_slices, positive_indexes = utils.Ct.read_and_create_from_image(
                series_uid
            ).get_sample_size()

            if self.use_full_ct:
                self.series_uid__slice_index += [
                    (series_uid, slice_ndx) for slice_ndx in range(n_ct_slices)
                ]
            else:
                self.series_uid__slice_index += [
                    (series_uid, slice_ndx) for slice_ndx in positive_indexes
                ]

        self.positive_candidates_info = [
            candidate for candidate in self.candidates_info if candidate.is_nodule
        ]

        _log.info("%s", repr(self))

    def __len__(self) -> int:
        return (
            self.training_length
            if self.training_length
            else len(self.series_uid__slice_index)
        )

    def __getitem__(self, index: int) -> dto.LunaSegmentationCandidate:
        if self.train:
            candidate_info = self.positive_candidates_info[
                index % len(self.positive_candidates_info)
            ]
            return self.get_cropped_ct_candidate(candidate_info)
        else:
            series_uid, slice_index = self.series_uid__slice_index[
                index % len(self.series_uid__slice_index)
            ]
            return self.get_full_ct_candidate(
                series_uid=series_uid, slice_index=slice_index
            )

    def get_full_ct_candidate(
        self, series_uid: str, slice_index: int
    ) -> dto.LunaSegmentationCandidate:
        ct_scan = utils.Ct.read_and_create_from_image(series_uid)
        candidate = torch.zeros((self.n_context_slices * 2 + 1, 512, 512))

        start_index = slice_index - self.n_context_slices
        end_index = slice_index + self.n_context_slices + 1
        for i, context_index in enumerate(range(start_index, end_index)):
            # When we reach beyond the bounds of the ct_scan.ct_hounsfield, we duplicate the first or last slice.
            context_index = max(context_index, 0)
            context_index = min(context_index, ct_scan.ct_hounsfield.shape[0] - 1)
            ct_scan_slice = torch.from_numpy(
                ct_scan.ct_hounsfield[context_index].astype(np.float32)
            )
            candidate[i] = ct_scan_slice

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        candidate.clamp_(-1000, 1000)

        positive_candidate_mask = torch.from_numpy(
            ct_scan.positive_mask[slice_index]
        ).unsqueeze(0)

        return dto.LunaSegmentationCandidate(
            candidate=candidate,
            positive_candidate_mask=positive_candidate_mask,
            series_uid=ct_scan.series_uid,
            slice_index=slice_index,
        )

    def get_cropped_ct_candidate(
        self, candidate_info: dto.CandidateMalignancyInfo
    ) -> dto.LunaSegmentationCandidate:
        """Instead of the full CT slices, we're going to train on 64 x 64 crops around our positive candidates
        (the actually-a-nodule candidates). These 64 x 64 patches will be taken randomly from a 96 x 96 crop
        centered on the nodule. We will also include three slices of context in both directions as additional
        “channels” to our 2D segmentation.

        We believe the whole-slice training was unstable essentially due to a class-balancing issue.
        Since each nodule is so small compared to the whole CT slice, we were right back in
        a needle-in-a-haystack situation"""
        ct_cutout, positive_mask, center = utils.Ct.create_ct_and_get_raw_image(
            series_uid=candidate_info.series_uid,
            center=candidate_info.center,
            cutout_shape=dto.CoordinatesIRC(index=7, row=96, col=96),
        )
        positive_mask = positive_mask[3:4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_cutout_offset = torch.from_numpy(
            ct_cutout[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.float32)
        positive_mask_offset = torch.from_numpy(
            positive_mask[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.long)

        return dto.LunaSegmentationCandidate(
            candidate=ct_cutout_offset,
            positive_candidate_mask=positive_mask_offset,
            series_uid=candidate_info.series_uid,
            slice_index=center.index,
        )

    def __repr__(self) -> str:
        n_negative_candidates = len(self.candidates_info) - len(
            self.positive_candidates_info
        )
        _repr = (
            f"{self.__class__.__name__}("
            f"len={len(self.candidates_info)}, "
            f"training_len={self.training_length}, "
            f"positive_len={len(self.positive_candidates_info)}, "
            f"negative_len={n_negative_candidates}, "
            f"train={self.train}, "
            f"validation_stride={self.validation_stride})"
        )
        return _repr

    def shuffle_samples(self) -> None:
        random.shuffle(self.candidates_info)
        random.shuffle(self.positive_candidates_info)
