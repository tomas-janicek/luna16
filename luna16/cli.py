import typer

from luna16 import bootstrap, training

cli = typer.Typer()


@cli.command(name="train_luna_classification")
def train_luna_classification(
    epochs: int = 1,
    num_workers: int = 8,
    batch_size: int = 32,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry()
    training.LunaClassificationLauncher(
        num_workers=num_workers,
        registry=registry,
        training_length=training_length,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).fit(
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        momentum=0.99,
        conv_channels=8,
    )
    registry.close_all_services()


@cli.command(name="tune_luna_classification")
def tune_luna_classification(
    epochs: int = 1,
    num_workers: int = 8,
    validation_stride: int = 5,
    training_length: int | None = None,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry()
    training.LunaClassificationLauncher(
        num_workers=num_workers,
        registry=registry,
        training_length=training_length,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    ).tune_parameters(epochs=epochs)
    registry.close_all_services()


@cli.command(name="profile_luna_classification")
def profile_luna_classification(
    epochs: int = 1,
    num_workers: int = 8,
    batch_size: int = 32,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Classification"
    registry = bootstrap.create_registry()
    luna_launcher = training.LunaClassificationLauncher(
        num_workers=num_workers,
        registry=registry,
        training_length=training_length,
        validation_stride=validation_stride,
        training_name=training_name,
        validation_cadence=5,
    )
    luna_launcher.profile_model(
        batch_size=batch_size,
        lr=0.001,
        momentum=0.99,
        conv_channels=8,
    )
    registry.close_all_services()


@cli.command(name="train_luna_segmentation")
def train_luna_segmentation(
    epochs: int = 1,
    num_workers: int = 8,
    batch_size: int = 32,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Segmentation"
    registry = bootstrap.create_registry()
    training.LunaSegmentationLauncher(
        training_name=training_name,
        num_workers=num_workers,
        registry=registry,
        training_length=training_length,
        validation_stride=validation_stride,
    ).fit(
        epochs=epochs,
        batch_size=batch_size,
    )
    registry.close_all_services()


@cli.command(name="train_luna_malignant_classification")
def train_luna_malignant_classification(
    state_name: str,
    epochs: int = 1,
    num_workers: int = 8,
    batch_size: int = 32,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Malignant Classification"
    registry = bootstrap.create_registry()
    training.LunaMalignantClassificationLauncher(
        num_workers=num_workers,
        registry=registry,
        training_length=training_length,
        validation_stride=validation_stride,
        state_name=state_name,
        training_name=training_name,
    ).fit(
        epochs=epochs,
        batch_size=batch_size,
    )
    registry.close_all_services()


if __name__ == "__main__":
    cli()
