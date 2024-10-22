from luna16 import datasets, dto


def test_luna16_create_train_defaults() -> None:
    ratio = dto.LunaClassificationRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(train=True, ratio=ratio)
    assert len(luna) == 157700


def test_luna16_create_test_defaults() -> None:
    ratio = dto.LunaClassificationRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(train=False, ratio=ratio)
    assert len(luna) == 8301
