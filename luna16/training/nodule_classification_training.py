import torch

from luna16 import (
    batch_iterators,
    datasets,
    models,
    modules,
    services,
    trainers,
)


def luna_classification_launcher(
    epochs: int,
    batch_size: int,
    validation_stride: int,
    num_workers: int,
    training_name: str,
    registry: services.ServiceContainer,
    training_length: int | None = None,
) -> None:
    logger = registry.get_service(services.LogMessageHandler)
    hyperparameters = registry.get_service(services.Hyperparameters)
    hyperparameters.add_hyperparameters(
        {
            "epochs": epochs,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "training_length": training_length,
            "validation_stride": validation_stride,
            "learning_rate": 0.001,
            "momentum": 0.99,
            "validation_cadence": 5,
        }
    )
    batch_iterator = batch_iterators.BatchIteratorProvider(logger=logger)
    module = modules.LunaModel()
    model = models.NoduleClassificationModel(
        model=module,
        optimizer=torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.99),
        batch_iterator=batch_iterator,
        logger=logger,
    )
    trainer = trainers.Trainer(name=training_name, logger=logger)
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
