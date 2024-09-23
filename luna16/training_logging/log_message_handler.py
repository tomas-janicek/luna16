import logging

from luna16 import services

from . import log_messages

_log = logging.getLogger(__name__)


class LogMessageHandler:
    def __init__(
        self,
        registry: services.ServiceContainer,
        log_messages: log_messages.LogMessageHandlersConfig,
    ) -> None:
        self.registry = registry
        self.log_messages = log_messages

    def handle_message(self, log_message: log_messages.LogMessage) -> None:
        message_type = self.log_messages.get(type(log_message))
        if message_type is None:
            _log.warning(
                "Message %s was not precess because it is not configured in handlers.",
                message_type,
            )
            return
        for handler in message_type:
            try:
                _log.debug("Handling event %s with handler %s", log_message, handler)
                handler(message=log_message, registry=self.registry)
            except Exception as error:
                _log.exception(
                    "Exception handling event %s. Error %s", log_message, error
                )
                continue
