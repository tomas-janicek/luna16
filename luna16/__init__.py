import logging.config
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    PYTORCH_ENABLE_MPS_FALLBACK: bool
    NUMEXPR_MAX_THREADS: int
    PYTHONPATH: str

    ML_FLOW_URL: str
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING: bool = True

    LOGGING_LEVEL: str = "DEBUG"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CACHE_DIR: Path = BASE_DIR / "cache"
    PROFILING_DIR: Path = BASE_DIR / "profiling"
    DATA_DOWNLOADED_DIR: Path = BASE_DIR / "data_downloaded"
    MODELS_DIR: Path = BASE_DIR / "models"
    DEEP_LEARNING_STYLE: Path = BASE_DIR / "deeplearning.mplstyle"
    DATA_DIR: Path = BASE_DIR / "data"

    PRESENT_CANDIDATES_FILE: str = "present_candidates.csv"
    NUM_WORKERS: int = 8


settings = Settings()  # type: ignore

mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "[%(asctime)s][%(levelname)s]: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": settings.LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console"],
            "level": settings.LOGGING_LEVEL,
        },
    },
}
logging.config.dictConfig(config=LOGGING)

plt.style.use(settings.DEEP_LEARNING_STYLE)


__all__ = ["settings"]
