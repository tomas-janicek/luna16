import pytest

from luna16 import datasets, dto


def test_create_train_defaults() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=2)
    assert len(luna) == 3


def test_create_test_defaults() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(train=False, ratio=ratio, validation_stride=2)
    assert len(luna) == 3


def test_create_train_with_big_validation_stride() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    luna = datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=6)
    assert len(luna) == 4


def test_create_train_with_too_big_validation_stride() -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    with pytest.raises(ValueError):
        datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=200)


def test_create_train_without_malignant(set_candidates_without_malignant: None) -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    with pytest.raises(ValueError):
        datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=2)


def test_create_train_without_nodule(set_candidates_without_nodule: None) -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    with pytest.raises(ValueError):
        datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=2)


def test_create_train_without_benign(set_candidates_without_benign: None) -> None:
    ratio = dto.NoduleRatio(positive=1, negative=1)
    with pytest.raises(ValueError):
        datasets.CutoutsDataset(train=True, ratio=ratio, validation_stride=2)
