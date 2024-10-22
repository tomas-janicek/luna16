from luna16 import augmentations, dto

from .nodule_classification_rationed import MalignantLunaDataset
from .nodule_cutouts import LunaCutoutsDataset
from .nodule_segmentation import LunaSegmentationDataset


def create_pre_configured_luna_cutouts(
    validation_stride: int,
) -> tuple[LunaCutoutsDataset, LunaCutoutsDataset]:
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
    train = LunaCutoutsDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    validation = LunaCutoutsDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    return train, validation


def create_pre_configured_luna_malignant(
    validation_stride: int,
    sort_by_series_uid: bool = True,
) -> tuple[MalignantLunaDataset, MalignantLunaDataset]:
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
    train = MalignantLunaDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    validation = MalignantLunaDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        transformations=transformations,
        filters=filters,
    )
    if sort_by_series_uid:
        train.sort_by_series_uid()
        validation.sort_by_series_uid()
    return train, validation


def create_pre_configured_luna_segmentation(
    validation_stride: int,
) -> tuple[LunaSegmentationDataset, LunaSegmentationDataset]:
    train = LunaSegmentationDataset(
        train=True,
        validation_stride=validation_stride,
    )
    validation = LunaSegmentationDataset(
        train=False,
        validation_stride=validation_stride,
    )
    return train, validation
