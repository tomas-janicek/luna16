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
        for handler in self.log_messages[type(log_message)]:
            try:
                _log.debug("Handling event %s with handler %s", log_message, handler)
                handler(message=log_message, registry=self.registry)
            except Exception:
                _log.exception("Exception handling event %s", log_message)
                continue
