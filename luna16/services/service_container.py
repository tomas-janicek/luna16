import typing

T = typing.TypeVar("T")


class ServiceContainer:
    def __init__(self) -> None:
        self.services = {}
        self.creators = {}
        self.closers = {}

    def register_service(
        self,
        type: typing.Type[T],
        value: T,
        on_registry_close: typing.Callable[..., None] | None = None,
    ) -> None:
        self.services[type] = value
        if on_registry_close:
            self.closers[type] = on_registry_close

    def register_creator(
        self,
        type: typing.Type[T],
        creator: typing.Callable[..., T],
        on_registry_close: typing.Callable[..., None] | None = None,
    ) -> None:
        self.creators[type] = creator
        if on_registry_close:
            self.closers[type] = on_registry_close

    def call_all_creators(self, **kwargs: typing.Any) -> None:
        for type, creator in self.creators.items():
            self.services[type] = creator(**kwargs)

    def get_service(self, type: typing.Type[T]) -> T:
        return self.services[type]

    def close_all_services(self) -> None:
        for type, closer in self.closers.items():
            service = self.services[type]
            closer(service)
