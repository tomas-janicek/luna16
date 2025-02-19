import os
import typing

import numpy as np
import typer
from ray import tune

from luna16 import bootstrap, data_processing, dto, enums, settings, training
from luna16.bootstrap import configurations

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


@cli.command(name="experiment")
def experiment() -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(
        configurations.BestCnnModel(n_blocks=4, dropout_rate=0.2),
        configurations.BestOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)),
        configurations.BestScheduler(gamma=0.1),
        dto.NoduleRatio(positive=1, negative=5),
    )
    training.NoduleClassificationLauncher(
        registry=registry,
        validation_stride=5,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        epochs=1,
        batch_size=128,
        version="0.0.0-experiment",
        log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
    )
    registry.close_all_services()


@cli.command(name="train_luna_classification")
def train_luna_classification(
    version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry(
        configurations.BestCnnModel(n_blocks=4, dropout_rate=0.2),
        configurations.BestOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)),
        configurations.BestScheduler(gamma=0.1),
        dto.NoduleRatio(positive=1, negative=5),
    )
    training.NoduleClassificationLauncher(
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
        configurations.BestCnnLoadedModel(
            n_blocks=4,
            name=from_name,
            version=from_version,
            finetune=False,
            model_loader=model_loader,
        ),
        configurations.BestOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)),
        configurations.BestScheduler(gamma=0.1),
        dto.NoduleRatio(positive=1, negative=5),
    )
    training.NoduleClassificationLauncher(
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


@cli.command(name="train_luna_malignant_classification")
def train_luna_malignant_classification(
    version: str,
    epochs: int = 1,
    batch_size: int = 64,
    validation_stride: int = 5,
    profile: bool = False,
) -> None:
    training_name = "Malignant Classification"
    registry = bootstrap.create_registry(
        configurations.BestCnnModel(n_blocks=4, dropout_rate=0.5),
        configurations.BestOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)),
        configurations.BestScheduler(gamma=0.1),
        dto.MalignantRatio(malignant=1, benign=5, not_module=10),
    )
    training.MalignantClassificationLauncher(
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
        configurations.BestCnnLoadedModel(
            n_blocks=4,
            name=from_name,
            version=from_version,
            finetune=True,
            model_loader=model_loader,
        ),
        configurations.BestOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)),
        configurations.BestScheduler(gamma=0.1),
        dto.MalignantRatio(malignant=1, benign=5, not_module=10),
    )
    training.MalignantClassificationLauncher(
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
    epochs: int = 10,
    validation_stride: int = 5,
) -> None:
    training_name = "Classification"

    hyperparameters: dict[str, typing.Any] = {
        "batch_size": tune.grid_search([64, 128, 256]),
        "learning_rate": tune.grid_search([0.00001, 0.0001, 0.001]),
        "scheduler_gamma": tune.grid_search([0.1, 0.5, 0.9]),
        # "weight_decay": tune.grid_search([0.0001, 0.001, 0.01]),
        "weight_decay": tune.grid_search([0.001]),
        # "luna_blocks": tune.grid_search([4, 8, 16]),
        "luna_blocks": tune.grid_search([4]),
        # "dropout_rate": tune.grid_search([0.1, 0.25, 0.3, 0.35, 0.4]),
        "dropout_rate": tune.grid_search([0.3]),
        "classification_ratio": tune.grid_search(
            [
                dto.NoduleRatio(positive=1, negative=5),
                dto.NoduleRatio(positive=1, negative=1),
            ]
        ),
    }

    def classification_tunning(config: dict[str, typing.Any]) -> float | np.float32:
        registry = bootstrap.create_tunning_registry(
            configurations.BestCnnModel(
                n_blocks=config["luna_blocks"], dropout_rate=config["dropout_rate"]
            ),
            configurations.BestOptimizer(
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
                betas=(0.9, 0.999),
            ),
            configurations.BestScheduler(gamma=config["scheduler_gamma"]),
        )
        scores = training.NoduleClassificationLauncher(
            registry=registry,
            validation_stride=validation_stride,
            training_name=training_name,
            validation_cadence=5,
        ).fit(
            version="tune",
            epochs=epochs,
            batch_size=config["batch_size"],
            log_every_n_examples=settings.LOG_EVERY_N_EXAMPLES,
        )
        registry.close_all_services()
        return scores["score"]

    tuner = tune.Tuner(classification_tunning, param_space=hyperparameters)
    tuner.fit()


if __name__ == "__main__":
    cli()
