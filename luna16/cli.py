import os

import typer

from luna16 import bootstrap, data_processing, enums, settings, training

cli = typer.Typer()


@cli.command(name="create_cutouts")
def create_cutouts(training_length: int | None = None) -> None:
    # If num workers is greater than 0, run in parallel
    cutout_service = data_processing.CtCutoutService()
    if settings.NUM_WORKERS:
        cutout_service.create_cutouts_concurrent(training_length=training_length)
    # Otherwise, run sequentially
    else:
        cutout_service.create_cutouts(training_length=training_length)


@cli.command(name="get_recommended_num_workers")
def get_recommended_num_workers() -> None:
    max_workers = min(32, (os.cpu_count() or 1))
    print(f"Recommendation for max number of workers is {max_workers}. ")


@cli.command(name="train_luna_classification")
def train_luna_classification(
    version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(enums.ConvModel())
    training.LunaClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        epochs=epochs,
        batch_size=batch_size,
        profile=profile,
        version=version,
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


@cli.command(name="train_luna_classification_slower")
def train_luna_classification_slower(
    version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(
        enums.ConvModel(),
        enums.OptimizerType.SLOWER_ADAM,
        enums.SchedulerType.SLOWER_STEP,
    )
    training.LunaClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        epochs=epochs,
        batch_size=batch_size,
        profile=profile,
        version=version,
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


@cli.command(name="load_train_luna_classification")
def load_train_luna_classification(
    version: str,
    model_loader: enums.ModelLoader,
    from_name: str,
    from_version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(
        enums.ConvLoadedModel(
            name=from_name,
            version=from_version,
            finetune=False,
            model_loader=model_loader,
        )
    )
    training.LunaClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        version=version,
        epochs=epochs,
        batch_size=batch_size,
        profile=profile,
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


@cli.command(name="tune_luna_classification")
def tune_luna_classification(
    epochs: int = 1,
    validation_stride: int = 5,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(enums.ConvModel())
    training.LunaClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).tune_parameters(epochs=epochs)
    registry.close_all_services()


@cli.command(name="train_luna_malignant_classification")
def train_luna_malignant_classification(
    version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Malignant Classification"
    registry = bootstrap.create_registry(enums.ConvModel())
    training.LunaMalignantClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        version=version,
        epochs=epochs,
        batch_size=batch_size,
        profile=profile,
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


@cli.command(name="load_train_luna_malignant_classification")
def load_train_luna_malignant_classification(
    version: str,
    model_loader: enums.ModelLoader,
    from_name: str,
    from_version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Malignant Classification"
    registry = bootstrap.create_registry(
        enums.ConvLoadedModel(
            name=from_name,
            version=from_version,
            finetune=True,
            model_loader=model_loader,
        )
    )
    training.LunaMalignantClassificationLauncher(
        registry=registry,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        version=version,
        epochs=epochs,
        batch_size=batch_size,
        profile=profile,
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


if __name__ == "__main__":
    cli()
