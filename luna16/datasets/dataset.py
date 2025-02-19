from luna16 import augmentations, dto

from .malignant_cutout import MalignantCutoutsDataset
from .nodule_cutouts import CutoutsDataset


def create_pre_configured_luna_cutouts(
    validation_stride: int, ratio: dto.NoduleRatio
) -> tuple[CutoutsDataset, CutoutsDataset]:
    transformations: list[augmentations.Transformation] = [
        augmentations.Flip(),
        augmentations.Offset(offset=0.1),
        augmentations.Scale(scale=0.2),
        augmentations.Rotate(),
    ]
    filters: list[augmentations.Filter] = [
        augmentations.Noise(noise=25.0),
    ]
    train = CutoutsDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    validation = CutoutsDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    return train, validation


def create_pre_configured_luna_malignant(
    validation_stride: int, ratio: dto.MalignantRatio
) -> tuple[MalignantCutoutsDataset, MalignantCutoutsDataset]:
    transformations: list[augmentations.Transformation] = [
        augmentations.Flip(),
        augmentations.Offset(offset=0.1),
        augmentations.Scale(scale=0.2),
        augmentations.Rotate(),
    ]
    filters: list[augmentations.Filter] = [
        augmentations.Noise(noise=25.0),
    ]
    train = MalignantCutoutsDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    validation = MalignantCutoutsDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    return train, validation
