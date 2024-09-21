import torch

from luna16 import (
    augmentations,
    batch_iterators,
    datasets,
    enums,
    models,
    modules,
    trainers,
    training_logging,
)


def luna_segmentation_launcher(
    epochs: int,
    batch_size: int,
    validation_stride: int,
    num_workers: int,
    training_name: str = "Segmentation",
    training_length: int | None = None,
) -> None:
    segmentation_logger = training_logging.SegmentationLoggingAdapter(
        training_name=training_name
    )
    batch_iterator = batch_iterators.BatchIteratorProvider(
        batch_loggers=segmentation_logger.batch_loggers
    )
    augmentation_model = augmentations.SegmentationAugmentation(
        flip=True, offset=0.03, scale=0.2, rotate=True, noise=25.0
    )
    module = modules.UNetNormalized(
        in_channels=7,
        n_classes=1,
        depth=3,
        wf=4,
        padding=True,
        batch_norm=True,
        up_mode=enums.UpMode.UP_CONV,
    )
    model = models.NoduleSegmentationModel(
        model=module,
        optimizer=torch.optim.Adam(module.parameters()),
        batch_iterator=batch_iterator,
        classification_logger=segmentation_logger,
        augmentation_model=augmentation_model,
    )
    trainer = trainers.Trainer(logger=segmentation_logger)
    train, validation = datasets.create_pre_configured_luna_segmentation(
        validation_stride=validation_stride,
        training_length=training_length,
    )
    data_module = datasets.DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train=train,
        validation=validation,
    )
    trainer.fit(model=model, epochs=epochs, data_module=data_module)
