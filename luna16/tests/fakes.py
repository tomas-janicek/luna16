from torch.utils import data as data_utils

from luna16 import message_handler


class FakeMessageHandler(message_handler.BaseMessageHandler):
    def __init__(self) -> None:
        self.requested_messages: list[message_handler.Message] = []

    def handle_message(self, message: message_handler.Message) -> None:
        self.requested_messages.append(message)


class FakeDataset(data_utils.Dataset[tuple[int, int]]):
    def __init__(self, n: int) -> None:
        self.numbers = list(range(n))
        self.numbers_reversed = [*self.numbers]
        self.numbers_reversed.reverse()

    def __len__(self) -> int:
        return len(self.numbers)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return self.numbers[index], self.numbers_reversed[index]
