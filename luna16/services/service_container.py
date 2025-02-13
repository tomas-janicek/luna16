import logging
import typing

_log = logging.getLogger(__name__)

T = typing.TypeVar("T")

ServiceType = type
Service = typing.Any
CreatorCallable = typing.Callable[..., Service]
ClosingCallable = typing.Callable[[Service], None]


class ServiceContainer:
    def __init__(self) -> None:
        self.services: dict[ServiceType, Service] = {}
        self.creators: dict[ServiceType, CreatorCallable] = {}
        self.closers: dict[ServiceType, ClosingCallable] = {}

    def register_service(
        self,
        type: type[T],
        value: T,
        on_registry_close: typing.Callable[..., None] | None = None,
    ) -> None:
        self.services[type] = value
        if on_registry_close:
            self.closers[type] = on_registry_close

    def register_creator(
        self,
        type: type[T],
        creator: typing.Callable[..., T],
        on_registry_close: typing.Callable[..., None] | None = None,
    ) -> None:
        self.creators[type] = creator
        if on_registry_close:
            self.closers[type] = on_registry_close

    def call_all_creators(self, **kwargs: typing.Any) -> None:
        for type, creator in self.creators.items():
            self.services[type] = creator(**kwargs)

    def get_service(self, type: type[T]) -> T:
        return self.services[type]

    def close_all_services(self) -> None:
        for type, closer in self.closers.items():
            service = self.services.get(type)
            if not service:
                _log.info(
                    "Service with type %s is not running and can not be closed.",
                    type,
                )
            closer(service)
