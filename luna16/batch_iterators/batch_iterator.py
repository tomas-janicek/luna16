import time
import typing

from torch.utils import data as data_utils
from tqdm import tqdm

from luna16 import enums, message_handler

from . import base

T = typing.TypeVar("T")


class BatchIteratorProvider(base.BaseIteratorProvider):
    def __init__(
        self,
        logger: message_handler.MessageHandler,
        logging_backoff: int = 5,
    ) -> None:
        self.logger = logger
        self.logging_backoff = logging_backoff

    def enumerate_batches(
        self, enumerable: data_utils.DataLoader[T], *, epoch: int, mode: enums.Mode
    ) -> typing.Iterator[tuple[int, T]]:
        batch_size = len(enumerable)

        log_bach_start = message_handler.LogBatchStart(
            epoch=epoch, mode=mode, batch_size=batch_size
        )
        self.logger.handle_message(log_bach_start)

        started_at = time.time()
        for current_index, item in tqdm(
            enumerate(enumerable),
            total=batch_size,
        ):
            yield (current_index, item)
            if current_index % self.logging_backoff == 0:
                log_start = message_handler.LogBatch(
                    epoch=epoch,
                    mode=mode,
                    batch_size=batch_size,
                    batch_index=current_index,
                    started_at=started_at,
                )
                self.logger.handle_message(log_start)

        log_bach_end = message_handler.LogBatchEnd(
            epoch=epoch, mode=mode, batch_size=batch_size
        )
        self.logger.handle_message(log_bach_end)
