import typer

from luna16 import services, training

cli = typer.Typer()


@cli.command(name="train_luna_classification")
def train_luna_classification(
    epochs: int = 1,
    num_workers: int = 8,
    batch_size: int = 32,
    training_length: int | None = None,
    validation_stride: int = 20,
) -> None:
    training_name = "Classification"
    registry = services.create_registry()
    training.luna_classification_launcher(
        epochs=epochs,
        num_workers=num_workers,
        registry=registry,
        batch_size=batch_size,
        training_length=training_length,
        validation_stride=validation_stride,
        training_name=training_name,
    )
    registry.close_all_services()


@cli.command(name="train_luna_segmentation")
def train_luna_segmentation(
    num_workers: int = 8,
    batch_size: int = 32,
    epochs: int = 1,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Segmentation"
    registry = services.create_registry()
    training.luna_segmentation_launcher(
        epochs=epochs,
        num_workers=num_workers,
        registry=registry,
        batch_size=batch_size,
        training_length=training_length,
        validation_stride=validation_stride,
        training_name=training_name,
    )
    registry.close_all_services()


@cli.command(name="train_luna_malignant_classification")
def train_luna_malignant_classification(
    state_name: str,
    num_workers: int = 8,
    batch_size: int = 32,
    epochs: int = 1,
    training_length: int | None = None,
    validation_stride: int = 5,
) -> None:
    training_name = "Malignant Classification"
    registry = services.create_registry()
    training.luna_malignant_classification_launcher(
        epochs=epochs,
        num_workers=num_workers,
        registry=registry,
        batch_size=batch_size,
        training_length=training_length,
        validation_stride=validation_stride,
        state_name=state_name,
        training_name=training_name,
    )
    registry.close_all_services()


if __name__ == "__main__":
    cli()
