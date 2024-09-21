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

    def handle(self, message: log_messages.LogMessage) -> None:
        self.queue = [message]
        while self.queue:
            message = self.queue.pop(0)
            self.handle_event(log_message=message, registry=self.registry)

    def handle_event(
        self,
        log_message: log_messages.LogMessage,
        registry: services.ServiceContainer,
    ) -> None:
        for handler in self.log_messages[type(log_message)]:
            try:
                _log.debug("Handling event %s with handler %s", log_message, handler)
                handler(message=log_message, registry=registry)
            except Exception:
                _log.exception("Exception handling event %s", log_message)
                continue
