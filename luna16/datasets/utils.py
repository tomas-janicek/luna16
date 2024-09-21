from luna16 import augmentations, dto

from .nodule_classification_rationed import LunaRationedDataset
from .nodule_segmentation import LunaSegmentationDataset


def create_pre_configured_luna_rationed(
    validation_stride: int,
    training_length: int | None = None,
) -> tuple[LunaRationedDataset, LunaRationedDataset]:
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
    train = LunaRationedDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    validation = LunaRationedDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    return train, validation


def create_pre_configured_luna_segmentation(
    validation_stride: int,
    training_length: int | None = None,
) -> tuple[LunaSegmentationDataset, LunaSegmentationDataset]:
    train = LunaSegmentationDataset(
        train=True,
        validation_stride=validation_stride,
        training_length=training_length,
    )
    validation = LunaSegmentationDataset(
        train=False,
        validation_stride=validation_stride,
        training_length=training_length,
    )
    return train, validation
