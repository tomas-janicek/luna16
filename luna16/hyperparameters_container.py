import typing


class HyperparameterContainer:
    def __init__(self) -> None:
        self.hyperparameters = {}

    def add_hyperparameter(self, name: str, value: typing.Any) -> None:
        self.hyperparameters[name] = value

    def add_hyperparameters(self, parameters: dict[str, typing.Any]) -> None:
        self.hyperparameters = self.hyperparameters | parameters

    def get_hyperparameter(self, name: str) -> typing.Any:
        return self.hyperparameters[name]
