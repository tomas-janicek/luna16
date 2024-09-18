import typer

from luna16 import augmentations, datasets, dto, training, training_logging

cli = typer.Typer()


@cli.command(name="train_luna_classification")
def train_luna_classification(
    num_workers: int = 8,
    batch_size: int = 32,
    epochs: int = 1,
    training_length: int | None = None,
    validation_stride: int = 20,
) -> None:
    training_name = "Classification"
    classification_logger = training_logging.ClassificationLoggingAdapter(
        training_name=training_name
    )
    luna_api = training.LunaTrainingAPI.create_with_optimizer_and_model(
        training_name=training_name,
        model_name="classification",
        num_workers=num_workers,
        batch_size=batch_size,
        classification_logger=classification_logger,
    )
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
    train = datasets.LunaRationedDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    validation = datasets.LunaRationedDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    luna_api.start_training(epochs=epochs, train=train, validation=validation)


@cli.command(name="train_luna_malignant_classification")
def train_luna_malignant_classification(
    state_name: str,
    num_workers: int = 8,
    batch_size: int = 32,
    epochs: int = 1,
    training_length: int | None = None,
    validation_stride: int = 20,
) -> None:
    training_name = "Malignant Classification"
    classification_logger = training_logging.ClassificationLoggingAdapter(
        training_name=training_name
    )
    luna_api = training.LunaTrainingAPI.create_from_saved_state(
        training_name=training_name,
        model_name="malignant",
        loaded_model_name="classification",
        state_name=state_name,
        num_workers=num_workers,
        batch_size=batch_size,
        classification_logger=classification_logger,
    )
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
    train = datasets.MalignantLunaDataset(
        ratio=ratio,
        train=True,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    validation = datasets.MalignantLunaDataset(
        ratio=ratio,
        train=False,
        validation_stride=validation_stride,
        training_length=training_length,
        transformations=transformations,
        filters=filters,
    )
    luna_api.start_training(epochs=epochs, train=train, validation=validation)


@cli.command(name="train_luna_segmentation")
def train_luna_segmentation(
    num_workers: int = 8,
    batch_size: int = 32,
    epochs: int = 1,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Segmentation"
    segmentation_logger = training_logging.SegmentationLoggingAdapter(
        training_name=training_name
    )
    augmentation_model = augmentations.SegmentationAugmentation(
        flip=True, offset=0.03, scale=0.2, rotate=True, noise=25.0
    )
    segmentation_api = training.SegmentationTrainingAPI.create_with_optimizer_and_model(
        num_workers=num_workers,
        batch_size=batch_size,
        augmentation_model=augmentation_model,
        segmentation_logger=segmentation_logger,
        training_name=training_name,
    )
    train = datasets.LunaSegmentationDataset(
        train=True,
        validation_stride=validation_stride,
        training_length=training_length,
    )
    validation = datasets.LunaSegmentationDataset(
        train=False,
        validation_stride=validation_stride,
        training_length=training_length,
    )
    segmentation_api.start_training(epochs=epochs, train=train, validation=validation)


if __name__ == "__main__":
    cli()
