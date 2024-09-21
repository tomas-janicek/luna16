import logging

import svcs

from . import log_messages

_log = logging.getLogger(__name__)


class LogMessageHandler:
    def __init__(
        self,
        registry: svcs.Registry,
        log_messages: log_messages.LogMessageHandlersConfig,
    ) -> None:
        self.registry = registry
        self.log_messages = log_messages

    def handle(self, message: log_messages.LogMessage) -> None:
        self.queue = [message]
        with svcs.Container(self.registry) as container:
            while self.queue:
                message = self.queue.pop(0)
                self.handle_event(log_message=message, container=container)

    def handle_event(
        self, log_message: log_messages.LogMessage, container: svcs.Container
    ) -> None:
        for handler in self.log_messages[type(log_message)]:
            try:
                _log.debug("Handling event %s with handler %s", log_message, handler)
                handler(message=log_message, services=container)
            except Exception:
                _log.exception("Exception handling event %s", log_message)
                continue
