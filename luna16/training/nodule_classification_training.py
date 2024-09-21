import torch

from luna16 import (
    batch_iterators,
    datasets,
    models,
    modules,
    trainers,
    training_logging,
)


def luna_classification_launcher(
    epochs: int,
    batch_size: int,
    validation_stride: int,
    num_workers: int,
    training_name: str = "Classification",
    training_length: int | None = None,
) -> None:
    classification_logger = training_logging.ClassificationLoggingAdapter(
        training_name=training_name
    )
    batch_iterator = batch_iterators.BatchIteratorProvider(
        batch_loggers=classification_logger.batch_loggers
    )
    module = modules.LunaModel()
    model = models.NoduleClassificationModel(
        model=module,
        optimizer=torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.99),
        batch_iterator=batch_iterator,
        classification_logger=classification_logger,
    )
    trainer = trainers.Trainer(logger=classification_logger)
    train, validation = datasets.create_pre_configured_luna_rationed(
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
