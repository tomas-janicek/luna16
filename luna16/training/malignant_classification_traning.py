from torch.profiler import schedule

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    message_handler,
    models,
    scoring,
    services,
)

from . import trainers


class MalignantClassificationLauncher:
    def __init__(
        self,
        validation_stride: int,
        training_name: str,
        registry: "services.ServiceContainer",
        validation_cadence: int,
    ) -> None:
        self.training_name = training_name
        self.validation_stride = validation_stride
        self.validation_cadence = validation_cadence
        self.registry = registry
        self.logger = registry.get_service(message_handler.MessageHandler)
        self.batch_iterator = batch_iterators.BatchIteratorProvider(logger=self.logger)

    def fit(
        self,
        version: str,
        epochs: int,
        batch_size: int,
        log_every_n_examples: int,
        profile: bool = False,
    ) -> scoring.PerformanceMetrics:
        module = self.registry.get_service(services.ClassificationModel)
        optimizer = self.registry.get_service(services.ClassificationOptimizer)
        lr_scheduler = self.registry.get_service(services.ClassificationScheduler)
        ratio = self.registry.get_service(dto.MalignantRatio)

        model = models.NoduleClassificationModel(
            module=module,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_iterator=self.batch_iterator,
            logger=self.logger,
            validation_cadence=self.validation_cadence,
            log_every_n_examples=log_every_n_examples,
        )
        trainer = trainers.Trainer[dto.LunaClassificationCandidate](
            name=self.training_name, version=version, logger=self.logger
        )
        train, validation = datasets.create_pre_configured_luna_malignant(
            validation_stride=self.validation_stride, ratio=ratio
        )
        data_module = datasets.DataModule(
            batch_size=batch_size,
            train=train,
            validation=validation,
        )
        if profile:
            tracing_schedule = schedule(
                skip_first=1, wait=1, warmup=1, active=1, repeat=4
            )
            return trainer.fit_profile(
                model=model,
                epochs=epochs,
                data_module=data_module,
                tracing_schedule=tracing_schedule,
            )
        return trainer.fit(model=model, epochs=epochs, data_module=data_module)
