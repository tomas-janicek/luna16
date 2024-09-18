from luna16 import datasets


def test_luna16_create_train_defaults() -> None:
    luna = datasets.LunaDataset(train=True)
    assert len(luna) == 157700


def test_luna16_create_test_defaults() -> None:
    luna = datasets.LunaDataset(train=False)
    assert len(luna) == 8301


def test_luna16_create_from_seriesuid_defaults() -> None:
    series_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.557875302364105947813979213632"
    luna = datasets.LunaDataset(train=True, series_uids=[series_uid])
    assert len(luna) == 1129


def test_get_item_from_luna() -> None:
    series_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.557875302364105947813979213632"
    luna = datasets.LunaDataset(train=True, series_uids=[series_uid])
    first_candidate = luna[0]
    assert first_candidate.labels[0] == 0
    assert first_candidate.labels[1] == 1
