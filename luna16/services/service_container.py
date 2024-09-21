import typing

T = typing.TypeVar("T")


class ServiceContainer:
    def __init__(self) -> None:
        self.services = {}
        self.creators = {}

    def register_service(self, type: typing.Type[T], value: T) -> None:
        self.services[type] = value

    def register_creator(
        self, type: typing.Type[T], creator: typing.Callable[..., T]
    ) -> None:
        self.creators[type] = creator

    def call_all_creators(self, **kwargs: typing.Any) -> None:
        for type, creator in self.creators:
            self.services[type] = creator(**kwargs)

    def get_service(self, type: typing.Type[T]) -> T:
        return self.services[type]

    def close_all_services(self) -> None: ...
