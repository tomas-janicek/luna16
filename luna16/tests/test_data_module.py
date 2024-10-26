from torch.utils.data import Dataset

from luna16 import datasets

from . import fakes


def test_data_module(
    fake_train_dataset: Dataset[fakes.SimpleCandidate],
    fake_validation_dataset: Dataset[fakes.SimpleCandidate],
) -> None:
    data_module = datasets.DataModule(
        batch_size=5, train=fake_train_dataset, validation=fake_validation_dataset
    )
    train_dataloader = data_module.get_dataloader(train=True)
    test_dataloader = data_module.get_dataloader(train=False)

    assert len(train_dataloader) == 2  # 10 / 5 = 2
    assert len(test_dataloader) == 1  # 5 / 5 = 5
