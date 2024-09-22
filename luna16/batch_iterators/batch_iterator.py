import time
import typing

from torch.utils import data as data_utils

from luna16 import enums, training_logging

from . import base

T = typing.TypeVar("T")


# TODO: implement tqdm and use it's progress bar
class BatchIteratorProvider(base.BaseIteratorProvider):
    def __init__(
        self,
        logger: training_logging.LogMessageHandler,
        logging_backoff: int = 4,
        max_length: int | None = None,
    ) -> None:
        self.logger = logger
        self.logging_backoff = logging_backoff
        self.max_length = max_length

    def enumerate_batches(
        self, enumerable: data_utils.DataLoader[T], *, epoch: int, mode: enums.Mode
    ) -> typing.Iterator[tuple[int, T]]:
        iteration_length = self.max_length or len(enumerable)

        log_bach_start = training_logging.LogBatchStart(
            epoch=epoch, mode=mode, batch_size=iteration_length
        )
        self.logger.handle_message(log_bach_start)

        started_at = time.time()
        for current_index, item in enumerate(enumerable):
            yield (current_index, item)
            if current_index % self.logging_backoff == 0:
                log_start = training_logging.LogBatch(
                    epoch=epoch,
                    mode=mode,
                    batch_size=iteration_length,
                    batch_index=current_index,
                    started_at=started_at,
                )
                self.logger.handle_message(log_start)

        log_bach_end = training_logging.LogBatchEnd(
            epoch=epoch, mode=mode, batch_size=iteration_length
        )
        self.logger.handle_message(log_bach_end)
