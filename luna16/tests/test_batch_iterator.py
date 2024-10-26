import torch
from torch.utils import data as data_utils
from torch.utils.data import DataLoader

from luna16 import batch_iterators, enums, message_handler


def test_batch_iterator(
    fake_message_handler: message_handler.BaseMessageHandler,
    fake_dataset: data_utils.Dataset[tuple[int, int]],
) -> None:
    batch_iterator = batch_iterators.BatchIteratorProvider(fake_message_handler)
    dataloader = DataLoader(fake_dataset, batch_size=5)

    batch_iter = batch_iterator.enumerate_batches(
        dataloader,
        epoch=0,
        mode=enums.Mode.TRAINING,
        candidate_batch_type=tuple[torch.Tensor, torch.Tensor],
    )
    i = None
    batch = []
    for i, batch in batch_iter:  # noqa: B007
        assert len(batch) == 2
        assert len(batch[0]) == 5
        assert len(batch[1]) == 5
    assert i == 1
