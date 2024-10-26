import logging
import typing

from . import messages

if typing.TYPE_CHECKING:
    from luna16 import services

_log = logging.getLogger(__name__)


class BaseMessageHandler(typing.Protocol):
    def handle_message(self, message: messages.Message) -> None: ...


class MessageHandler:
    def __init__(
        self,
        registry: "services.ServiceContainer",
        messages: messages.MessageHandlersConfig,
    ) -> None:
        self.registry = registry
        self.messages = messages

    def handle_message(self, message: messages.Message) -> None:
        message_type = self.messages.get(type(message))
        if message_type is None:
            _log.warning(
                "Message %s was not precess because it is not configured in handlers.",
                message_type,
            )
            return
        for handler in message_type:
            try:
                _log.debug("Handling event %s with handler %s", message, handler)
                handler(message=message, registry=self.registry)
            except Exception as error:
                _log.exception("Exception handling event %s. Error %s", message, error)
                continue
