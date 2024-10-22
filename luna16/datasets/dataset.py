from luna16 import augmentations, dto

from .nodule_cutouts import CutoutsDataset, MalignantCutoutsDataset


def create_pre_configured_luna_cutouts(
    validation_stride: int,
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
    ratio = dto.LunaClassificationRatio(positive=1, negative=1)
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
    validation_stride: int,
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
    ratio = dto.LunaMalignantRatio(benign=1, malignant=1, not_module=1)
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
