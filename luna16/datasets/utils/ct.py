import logging

import numpy as np
import SimpleITK as sitk
from diskcache import FanoutCache
from numpy import typing as np_typing

from luna16 import dto, enums
from luna16.settings import settings

from . import candidates

_log = logging.getLogger(__name__)


cache = FanoutCache(
    settings.CACHE_DIR / "luna16",
    shards=64,
    timeout=1,
    size_limit=3e11,
)


class Ct:
    def __init__(
        self,
        series_uid: str,
        ct_hounsfield: np_typing.NDArray[np.float32],
        origin: dto.CoordinatesXYZ,
        voxel_size: dto.CoordinatesXYZ,
        transformation_direction: np_typing.NDArray[np.float32],
        threshold_hounsfield: int = -700,
    ) -> None:
        self.series_uid = series_uid
        self.ct_hounsfield = ct_hounsfield
        self.origin = origin
        self.voxel_size = voxel_size
        self.transformation_direction = transformation_direction

        self.candidates = candidates.get_grouped_candidates_with_malignancy_info()[
            self.series_uid
        ]
        self.positive_candidates = [
            candidate for candidate in self.candidates if candidate.is_nodule
        ]
        self.threshold_hounsfield = threshold_hounsfield
        self.positive_mask = self.get_annotation_mask()
        self.positive_indexes: list[int] = (
            self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist()
        )

    @staticmethod
    @cache.memoize(typed=True)
    def create_ct_and_get_raw_image(
        series_uid: str,
        center: dto.CoordinatesXYZ,
        cutout_shape: dto.CoordinatesIRC,
    ) -> tuple[
        np_typing.NDArray[np.float32], np_typing.NDArray[np.bool_], dto.CoordinatesIRC
    ]:
        ct_scan: Ct = Ct.read_and_create_from_image(series_uid=series_uid)
        ct_chunk, positive_chunk, center_irc = ct_scan.get_ct_cutout_from_center(
            center, cutout_shape
        )
        return ct_chunk, positive_chunk, center_irc

    def get_annotation_mask(self) -> np_typing.NDArray[np.bool_]:
        bounding_box = np.zeros_like(self.ct_hounsfield, dtype=np.bool_)

        for candidate in self.positive_candidates:
            center = candidate.center.to_irc(
                origin=self.origin,
                voxel_size=self.voxel_size,
                transformation_direction=self.transformation_direction,
            )

            index_radius = self._get_bounding_radius_for_dimension(
                center=center, dimension=enums.DimensionIRC.INDEX
            )
            row_radius = self._get_bounding_radius_for_dimension(
                center=center, dimension=enums.DimensionIRC.ROW
            )
            col_radius = self._get_bounding_radius_for_dimension(
                center=center, dimension=enums.DimensionIRC.COL
            )

            assert index_radius > 0
            assert row_radius > 0
            assert col_radius > 0

            bounding_box[
                center.index - index_radius : center.index + index_radius + 1,
                center.row - row_radius : center.row + row_radius + 1,
                center.col - col_radius : center.col + col_radius + 1,
            ] = True

        mask = bounding_box & (self.ct_hounsfield > self.threshold_hounsfield)

        return mask

    @staticmethod
    def read_and_create_from_image(series_uid: str) -> "Ct":
        ct_scan_subsets = settings.DATA_DOWNLOADED_DIR / "ct_scan_subsets"
        ct_mhd_files = list(ct_scan_subsets.glob(f"subset*/{series_uid}.mhd"))
        if not ct_mhd_files:
            raise ValueError(
                f"The dataset does not contain CT scan with series UID {series_uid}."
            )
        else:
            ct_mhd_path = ct_mhd_files[0]

        ct_mhd_image = sitk.ReadImage(ct_mhd_path)
        # ct_hounsfield is a three-dimensional array. All three dimensions are spatial,
        # and the single intensity channel is implicit.
        ct_hounsfield = np.array(sitk.GetArrayFromImage(ct_mhd_image), dtype=np.float32)
        # We are clipping values to [-1000, 1000] because data above this range is not relevant
        # to finding nodules. More in data_exaploration.ipynb notebook.
        ct_hounsfield.clip(min=-1000, max=1000, out=ct_hounsfield)

        origin = dto.CoordinatesXYZ(*ct_mhd_image.GetOrigin())
        voxel_size = dto.CoordinatesXYZ(*ct_mhd_image.GetSpacing())
        transformation_direction: np_typing.NDArray[np.float32] = np.array(
            ct_mhd_image.GetDirection()
        ).reshape(3, 3)
        return Ct(
            series_uid=series_uid,
            ct_hounsfield=ct_hounsfield,
            origin=origin,
            voxel_size=voxel_size,
            transformation_direction=transformation_direction,
        )

    def get_sample_size(self) -> tuple[int, list[int]]:
        return self.ct_hounsfield.shape[0], self.positive_indexes

    def get_ct_cutout_from_center(
        self, center: dto.CoordinatesXYZ, cutout_shape: dto.CoordinatesIRC
    ) -> tuple[
        np_typing.NDArray[np.float32], np_typing.NDArray[np.bool_], dto.CoordinatesIRC
    ]:
        center_irc = center.to_irc(
            self.origin,
            self.voxel_size,
            self.transformation_direction,
        )

        cutout_slices: list[slice] = []
        for axis, center_coord in enumerate(center_irc):
            self._raise_if_center_out_of_bound(axis=axis, center_coord=center_coord)

            start_index, end_index = self._get_cutout_bounds(
                cutout_shape=cutout_shape, axis=axis, center_coord=center_coord
            )

            cutout_slices.append(slice(start_index, end_index))

        ct_cutout = self.ct_hounsfield[tuple(cutout_slices)]
        positive_chunk = self.positive_mask[tuple(cutout_slices)]
        return ct_cutout, positive_chunk, center_irc

    def _get_bounding_radius_for_dimension(
        self, center: dto.CoordinatesIRC, dimension: enums.DimensionIRC
    ) -> float:
        radius = 2
        center_increased = center.model_copy().move_dimension(
            dimension=dimension, move_by=radius
        )
        center_decreased = center.model_copy().move_dimension(
            dimension=dimension, move_by=-radius
        )

        try:
            while (
                self.ct_hounsfield[
                    center_increased.index, center_increased.row, center_increased.col
                ]
                > self.threshold_hounsfield
                and self.ct_hounsfield[
                    center_decreased.index,
                    center_decreased.row,
                    center_decreased.col,
                ]
                > self.threshold_hounsfield
            ):
                radius += 1
                center_increased = center.model_copy().move_dimension(
                    dimension=dimension, move_by=radius
                )
                center_decreased = center.model_copy().move_dimension(
                    dimension=dimension, move_by=-radius
                )
        except IndexError:
            return radius - 1
        return radius

    def _get_cutout_bounds(
        self, cutout_shape: dto.CoordinatesIRC, axis: int, center_coord: int
    ) -> tuple[int, int]:
        start_index = int(round(center_coord - cutout_shape[axis] / 2))
        end_index = int(start_index + cutout_shape[axis])

        if start_index < 0:
            _log.debug(
                "Out of bound when getting 'start' index of CT array "
                "with series UID %s. Axis: %d, Coordinate: %d.",
                self.series_uid,
                axis,
                center_coord,
            )
            start_index = 0
            end_index = int(cutout_shape[axis])

        if end_index > self.ct_hounsfield.shape[axis]:
            _log.debug(
                "Out of bound when getting 'end' index of CT array "
                "with series UID %s. Axis: %d, Coordinate: %d.",
                self.series_uid,
                axis,
                center_coord,
            )
            end_index = self.ct_hounsfield.shape[axis]
            start_index = int(self.ct_hounsfield.shape[axis] - cutout_shape[axis])

        return start_index, end_index

    def _raise_if_center_out_of_bound(self, *, axis: int, center_coord: int):
        assert (
            center_coord >= 0 and center_coord < self.ct_hounsfield.shape[axis]
        ), repr(
            [
                self.series_uid,
                self.origin,
                self.voxel_size,
                axis,
            ]
        )
