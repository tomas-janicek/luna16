import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import typing as np_typing

from luna16 import data_processing, datasets, dto

_log = logging.getLogger(__name__)

matplotlib.use("nbagg")
clim = (-1000.0, 300)


def show_positive_candidate() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(ratio=ratio)

    # When ration is 1/1, first candidate is always positive
    positive_candidate = luna[0]
    show_nodule_visualization(positive_candidate)


def show_negative_candidate() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(ratio=ratio)

    # When ration is 1/1, second candidate is always negative
    negative_candidate = luna[1]
    show_nodule_visualization(negative_candidate)


def show_nodule_visualization(  # noqa: C901
    candidate: dto.LunaClassificationCandidate,
) -> None:
    slice_of_ct, is_nodule_tensor, series_uid, center_irc = candidate
    one_ct_image = slice_of_ct[0].numpy()
    entire_ct_scan = data_processing.Ct.read_and_create_from_image(
        series_uid=series_uid
    )

    # Prepare matplotlib figure

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
    plt.imshow(one_ct_image[one_ct_image.shape[0] // 2], clim=clim, cmap="gray")

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title(f"Row {int(center_irc[1])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(one_ct_image[:, one_ct_image.shape[1] // 2], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title(f"Col {int(center_irc[2])}", fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(one_ct_image[:, :, one_ct_image.shape[2] // 2], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title(f"Slice {index}", fontsize=30)
            for label in subplot.get_xticklabels() + subplot.get_yticklabels():
                label.set_fontsize(20)
            plt.imshow(one_ct_image[index], clim=clim, cmap="gray")

    if is_nodule_tensor[1]:
        print(f"Showing nodule with series UID {series_uid}")
    else:
        print(f"Showing CT scan with series UID {series_uid} without nodule.")


def plot_cutout(
    center_irc: torch.Tensor, one_ct_image: np_typing.NDArray[np.floating]
) -> None:
    _fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

    ax1.set_title(f"Index {int(center_irc[0])}", fontsize=30)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(20)
    ax1.imshow(one_ct_image[one_ct_image.shape[0] // 2], clim=clim, cmap="gray")

    ax2.set_title(f"Row {int(center_irc[1])}", fontsize=30)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(20)
    ax2.imshow(one_ct_image[:, one_ct_image.shape[1] // 2], clim=clim, cmap="gray")

    ax3.set_title(f"Col {int(center_irc[2])}", fontsize=30)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontsize(20)
    ax3.imshow(one_ct_image[:, :, one_ct_image.shape[2] // 2], clim=clim, cmap="gray")
