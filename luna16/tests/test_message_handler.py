from dataclasses import dataclass

from luna16 import message_handler, services


@dataclass
class FakeMessage(message_handler.Message):
    text: str


def test_message_handler() -> None:
    processed_messages: list[message_handler.Message] = []

    def handle_fake_message(
        message: FakeMessage, registry: services.ServiceContainer
    ) -> None:
        processed_messages.append(message)

    registry = services.ServiceContainer()
    handler = message_handler.MessageHandler(
        registry=registry, messages={FakeMessage: (handle_fake_message,)}
    )
    message = FakeMessage(text="Test text")

    handler.handle_message(message=message)

    assert len(processed_messages) == 1
