import pytest
from torch import nn
from torch.utils import data as data_utils

from luna16 import datasets, message_handler, models, services, settings

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
def empty_registry() -> services.ServiceContainer:
    registry = services.ServiceContainer()
    return registry


@pytest.fixture
def fake_message_handler(
    empty_registry: services.ServiceContainer,
) -> message_handler.BaseMessageHandler:
    message_handler = fakes.FakeMessageHandler(registry=empty_registry)
    return message_handler


@pytest.fixture
def fake_dataset() -> data_utils.Dataset[fakes.SimpleCandidate]:
    fake_dataset = fakes.FakeDataset(n=10)
    return fake_dataset


@pytest.fixture
def fake_train_dataset() -> data_utils.Dataset[fakes.SimpleCandidate]:
    fake_dataset = fakes.FakeDataset(n=10)
    return fake_dataset


@pytest.fixture
def fake_validation_dataset() -> data_utils.Dataset[fakes.SimpleCandidate]:
    fake_dataset = fakes.FakeDataset(n=5)
    return fake_dataset


@pytest.fixture
def fake_module() -> nn.Module:
    fake_module = fakes.FakeModule(in_features=2, out_features=1)
    return fake_module


@pytest.fixture
def fake_model(fake_module: nn.Module) -> models.BaseModel[fakes.SimpleCandidate]:
    fake_model = fakes.FakeModel(module=fake_module)
    return fake_model


@pytest.fixture
def fake_data_module(
    fake_train_dataset: data_utils.Dataset[fakes.SimpleCandidate],
    fake_validation_dataset: data_utils.Dataset[fakes.SimpleCandidate],
) -> datasets.DataModule[fakes.SimpleCandidate]:
    fake_data_module = datasets.DataModule(
        batch_size=5, train=fake_train_dataset, validation=fake_validation_dataset
    )
    return fake_data_module
