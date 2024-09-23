import numpy as np
import torch
from numpy import typing as np_typing
from torch.utils import data as data_utils
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import datasets, dto, enums, services

from .. import log_messages, utils


def log_images_to_tensorboard(
    message: log_messages.LogImages[dto.LunaSegmentationCandidate],
    registry: services.ServiceContainer,
) -> None:
    message.model.eval()
    tensorboard_writer = utils.get_tensortboard_writer(
        mode=message.mode, registry=registry
    )

    # Always takes the first n CT scans so we can watch process of our training on them.
    first_12_series_uids = sorted(message.dataloader.dataset.series_uids)[:12]  # type: ignore
    for series_index, series_uid in enumerate(first_12_series_uids):
        ct_scan = datasets.Ct.read_and_create_from_image(series_uid)

        # Six slices are logged because TensorBoard can visualize 12 images on one page.
        # Thus we get 6 label slices and 6 prediction slices on one page
        # and we can compare them.
        for slice_index in range(6):
            prediction, label, ct_slice = get_data_from_ct(
                dataloader=message.dataloader,
                series_uid=series_uid,
                ct_scan=ct_scan,
                slice_index=slice_index,
                model=message.model,
                device=message.device,
            )
            log_slice(
                mode=message.mode,
                series_index=series_index,
                slice_ndx=slice_index,
                ct_slice=ct_slice,
                label=label,
                prediction=prediction,
                tensorboard_writer=tensorboard_writer,
                n_processed_samples=message.n_processed_samples,
            )
            if message.epoch == 1:
                log_ground_truth_slice(
                    mode=message.mode,
                    tensorboard_writer=tensorboard_writer,
                    series_index=series_index,
                    slice_ndx=slice_index,
                    ct_slice=ct_slice,
                    label=label,
                    n_processed_samples=message.n_processed_samples,
                )
            # This flush prevents TB from getting confused about which
            # data item belongs where.
            tensorboard_writer.flush()


def log_slice(
    *,
    tensorboard_writer: SummaryWriter,
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

    tensorboard_writer.add_image(
        tag=f"{mode.value}/{series_index}_prediction_{slice_ndx}",
        img_tensor=image,
        global_step=n_processed_samples,
        dataformats="HWC",
    )


def log_ground_truth_slice(
    *,
    mode: enums.Mode,
    n_processed_samples: int,
    series_index: int,
    slice_ndx: int,
    ct_slice: np_typing.NDArray[np.float32],
    label: np_typing.NDArray[np.bool_],
    tensorboard_writer: SummaryWriter,
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

    tensorboard_writer.add_image(
        tag=f"{mode.value}/{series_index}_label_{slice_ndx}",
        img_tensor=image,
        global_step=n_processed_samples,
        dataformats="HWC",
    )


def get_data_from_ct(
    dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
    series_uid: str,
    ct_scan: datasets.Ct,
    slice_index: int,
    model: torch.nn.Module,
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
