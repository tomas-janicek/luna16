import logging
import typing

import matplotlib

from luna16.datasets import utils

matplotlib.use("nbagg")

import matplotlib.pyplot as plt

from luna16 import datasets, dto

_log = logging.getLogger(__name__)

clim = (-1000.0, 300)


def get_positive_samples(
    start_at: int = 0, limit: int = 100, print_samples: bool = False
) -> list[tuple[int, dto.CandidateInfo]]:
    luna = datasets.LunaDataset()
    positive_samples: list[tuple[int, dto.CandidateInfo]] = []
    for i, candidate in enumerate(luna.candidates_info[start_at:]):
        if candidate.is_nodule:
            positive_samples.append((i, candidate))

            if print_samples:
                _log.info(len(positive_samples), candidate)

        if len(positive_samples) >= limit:
            break

    return positive_samples


def show_candidate(series_uid: str, **kwargs: typing.Any) -> None:
    luna = datasets.LunaDataset(series_uids=[series_uid], **kwargs)
    batch_index = get_first_positive_sample(luna)

    entire_ct_scan = utils.Ct.read_and_create_from_image(series_uid=series_uid)
    slice_of_ct, is_nodule_tensor, series_uid, center_irc = luna[batch_index]
    slice_of_ct_image = slice_of_ct[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    # Visualize entire CT scan

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title(f"Index {int(center_irc[0])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(entire_ct_scan.ct_hounsfield[int(center_irc[0])], clim=clim, cmap="gray")

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title(f"Row {int(center_irc[1])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(
        entire_ct_scan.ct_hounsfield[:, int(center_irc[1])], clim=clim, cmap="gray"
    )
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title(f"Col {int(center_irc[2])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(
        entire_ct_scan.ct_hounsfield[:, :, int(center_irc[2])], clim=clim, cmap="gray"
    )
    plt.gca().invert_yaxis()

    # Visualize selected nodule

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title(f"Index {int(center_irc[0])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(
        slice_of_ct_image[slice_of_ct_image.shape[0] // 2], clim=clim, cmap="gray"
    )

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title(f"Row {int(center_irc[1])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(
        slice_of_ct_image[:, slice_of_ct_image.shape[1] // 2], clim=clim, cmap="gray"
    )
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title(f"Col {int(center_irc[2])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(
        slice_of_ct_image[:, :, slice_of_ct_image.shape[2] // 2], clim=clim, cmap="gray"
    )
    plt.gca().invert_yaxis()

    # No idea what this does

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title(f"Slice {index}", fontsize=30)
            for label in subplot.get_xticklabels() + subplot.get_yticklabels():
                label.set_fontsize(20)
            plt.imshow(slice_of_ct_image[index], clim=clim, cmap="gray")

    print(series_uid, batch_index, bool(is_nodule_tensor[0]))


def get_first_positive_sample(luna: datasets.LunaDataset) -> int:
    positive_samples_indexes = [
        i for i, candidate in enumerate(luna.candidates_info) if candidate.is_nodule
    ]

    if positive_samples_indexes:
        first_positive_sample_index = positive_samples_indexes[0]
    else:
        print("Warning: no positive samples found; using first negative sample.")
        first_positive_sample_index = 0
    return first_positive_sample_index
