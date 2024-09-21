import typing

import numpy as np
import torch
from numpy import typing as np_typing
from torch import nn
from torch.utils import data as data_utils
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import enums
from luna16.datasets import utils

from .. import dto
from . import base


class CtImageLoggerWrapper(base.BaseLoggerWrapper):
    def __init__(self, training_name: str) -> None:
        self.training_name = training_name

    def open_logger(
        self,
        *,
        training_writer: SummaryWriter,
        validation_writer: SummaryWriter,
        **kwargs: typing.Any,
    ) -> None:
        self.training_writer = training_writer
        self.validation_writer = validation_writer

    def log_images(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        model: nn.Module,
        device: torch.device,
    ) -> None:
        model.eval()

        # Always takes the first n CT scans so we can watch process of our training on them.
        first_12_series_uids = sorted(dataloader.dataset.series_uids)[:12]  # type: ignore
        for series_index, series_uid in enumerate(first_12_series_uids):
            ct_scan = utils.Ct.read_and_create_from_image(series_uid)

            # Six slices are logged because TensorBoard can visualize 12 images on one page.
            # Thus we get 6 label slices and 6 prediction slices on one page
            # and we can compare them.
            for slice_index in range(6):
                prediction, label, ct_slice = self.get_data_from_ct(
                    dataloader=dataloader,
                    series_uid=series_uid,
                    ct_scan=ct_scan,
                    slice_index=slice_index,
                    model=model,
                    device=device,
                )
                self.log_slice(
                    mode=mode,
                    series_index=series_index,
                    slice_ndx=slice_index,
                    ct_slice=ct_slice,
                    label=label,
                    prediction=prediction,
                    n_processed_samples=n_processed_samples,
                )
                if epoch == 1:
                    self.log_ground_truth_slice(
                        mode=mode,
                        series_index=series_index,
                        slice_ndx=slice_index,
                        ct_slice=ct_slice,
                        label=label,
                        n_processed_samples=n_processed_samples,
                    )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                self.training_writer.flush()

    def log_slice(
        self,
        *,
        mode: enums.Mode,
        n_processed_samples: int,
        series_index: int,
        slice_ndx: int,
        ct_slice: np_typing.NDArray[np.float32],
        label: np_typing.NDArray[np.bool_],
        prediction: np_typing.NDArray[np.bool_],
    ) -> None:
        # Image is in this case 512 x 512 array with three channels (RGB).
        image = np.zeros((512, 512, 3), dtype=np.float32)
        # First channel represent CT scan
        image[:, :, :] = ct_slice.reshape((512, 512, 1))
        # False positives are flagged as red and overlaid on the image.
        image[:, :, 0] += prediction & (1 - label)
        # False negatives are orange.
        image[:, :, 0] += (1 - prediction) & label
        image[:, :, 1] += ((1 - prediction) & label) * 0.5
        # True positives are green.
        image[:, :, 1] += prediction & label
        image *= 0.5
        image.clip(0, 1, image)

        tensorboard_writer = self._get_writer(mode=mode)
        tensorboard_writer.add_image(
            tag=f"{mode.value}/{series_index}_prediction_{slice_ndx}",
            img_tensor=image,
            global_step=n_processed_samples,
            dataformats="HWC",
        )

    def log_ground_truth_slice(
        self,
        *,
        mode: enums.Mode,
        n_processed_samples: int,
        series_index: int,
        slice_ndx: int,
        ct_slice: np_typing.NDArray[np.float32],
        label: np_typing.NDArray[np.bool_],
    ) -> None:
        """We also want to save the ground truth that we're using to train,
        which will form the top row of our TensorBoard CT slices"""
        image = np.zeros((512, 512, 3), dtype=np.float32)
        image[:, :, :] = ct_slice.reshape((512, 512, 1))
        # Labels (the real nodule pixels) are green.
        image[:, :, 1] += label

        image *= 0.5
        image[image < 0] = 0
        image[image > 1] = 1

        tensorboard_writer = self._get_writer(mode=mode)
        tensorboard_writer.add_image(
            tag=f"{mode.value}/{series_index}_label_{slice_ndx}",
            img_tensor=image,
            global_step=n_processed_samples,
            dataformats="HWC",
        )

    def get_data_from_ct(
        self,
        dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        series_uid: str,
        ct_scan: utils.Ct,
        slice_index: int,
        model: nn.Module,
        device: torch.device,
    ) -> tuple[
        np_typing.NDArray[np.bool_],
        np_typing.NDArray[np.bool_],
        np_typing.NDArray[np.float32],
    ]:
        ct_ndx = slice_index * (ct_scan.ct_hounsfield.shape[0] - 1) // 5
        ct_candidate: dto.LunaSegmentationCandidate = (
            dataloader.dataset.get_full_ct_candidate(  # type: ignore
                series_uid, ct_ndx
            )
        )

        input = ct_candidate.candidate.to(device).unsqueeze(0)
        label = ct_candidate.positive_candidate_mask.to(device).unsqueeze(0)
        prediction = model(input)[0]
        prediction_array: np_typing.NDArray[np.bool_] = (
            prediction.to("cpu").detach().numpy()[0] > 0.5
        )
        label_array: np_typing.NDArray[np.bool_] = label.cpu().numpy()[0][0] > 0.5

        ct_candidate.candidate[:-1, :, :] /= 2000
        ct_candidate.candidate[:-1, :, :] += 0.5

        ct_slice: np_typing.NDArray[np.float32] = ct_candidate.candidate[
            dataloader.dataset.n_context_slices  # type: ignore
        ].numpy()

        return prediction_array, label_array, ct_slice

    def close_logger(self) -> None:
        self.training_writer.close()
        self.validation_writer.close()

    def _get_writer(self, mode: enums.Mode) -> SummaryWriter:
        match mode:
            case enums.Mode.TRAINING:
                tensorboard_writer = self.training_writer
            case enums.Mode.VALIDATING:
                tensorboard_writer = self.validation_writer
        return tensorboard_writer
