import pytest
from torch.utils import data as data_utils

from luna16 import message_handler, settings

from . import fakes


@pytest.fixture(autouse=True)
def set_testing_settings() -> None:
    settings.DATA_DIR = settings.BASE_DIR / "test_data"

    settings.CACHE_DIR = settings.BASE_DIR / "test_data" / "cache"
    settings.PRESENT_CANDIDATES_FILE = "present_candidates.csv"
    settings.DATA_DOWNLOADED_DIR = settings.BASE_DIR / "test_data" / "cts"
    settings.NUM_WORKERS = 0


@pytest.fixture
def set_candidates_without_malignant() -> None:
    settings.PRESENT_CANDIDATES_FILE = "present_candidates_without_malignant.csv"


@pytest.fixture
def set_candidates_without_nodule() -> None:
    settings.PRESENT_CANDIDATES_FILE = "present_candidates_without_nodule.csv"


@pytest.fixture
def set_candidates_without_benign() -> None:
    settings.PRESENT_CANDIDATES_FILE = "present_candidates_without_benign.csv"


@pytest.fixture
def fake_message_handler() -> message_handler.BaseMessageHandler:
    message_handler = fakes.FakeMessageHandler()
    return message_handler


@pytest.fixture
def fake_dataset() -> data_utils.Dataset[tuple[int, int]]:
    fake_dataset = fakes.FakeDataset(n=10)
    return fake_dataset
