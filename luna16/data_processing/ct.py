import logging

import numpy as np
import SimpleITK as sitk
from numpy import typing as np_typing

from luna16 import dto, settings

_log = logging.getLogger(__name__)


class Ct:
    def __init__(
        self,
        series_uid: str,
        ct_hounsfield: np_typing.NDArray[np.float32],
        origin: dto.CoordinatesXYZ,
        voxel_size: dto.CoordinatesXYZ,
        transformation_direction: np_typing.NDArray[np.float32],
    ) -> None:
        self.series_uid = series_uid
        self.ct_hounsfield = ct_hounsfield
        self.origin = origin
        self.voxel_size = voxel_size
        self.transformation_direction = transformation_direction

    @staticmethod
    def read_and_create_from_image(series_uid: str) -> "Ct":
        ct_mhd_files = list(settings.DATA_DOWNLOADED_DIR.glob(f"**/{series_uid}.mhd"))
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
            ct_mhd_image.GetDirection(), dtype=np.float32
        ).reshape(3, 3)

        return Ct(
            series_uid=series_uid,
            ct_hounsfield=ct_hounsfield,
            origin=origin,
            voxel_size=voxel_size,
            transformation_direction=transformation_direction,
        )

    def get_ct_cutout_from_center(
        self, center: dto.CoordinatesXYZ, cutout_shape: dto.CoordinatesIRC
    ) -> tuple[np_typing.NDArray[np.float32], dto.CoordinatesIRC]:
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
        return ct_cutout, center_irc

    def _get_cutout_bounds(
        self, cutout_shape: dto.CoordinatesIRC, axis: int, center_coord: np.int16
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

    def _raise_if_center_out_of_bound(
        self, *, axis: int, center_coord: np.int16
    ) -> None:
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
