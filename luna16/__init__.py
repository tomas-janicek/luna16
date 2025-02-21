import logging.config
from pathlib import Path

import mlflow
import pydantic_settings
from pydantic import computed_field


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

    @computed_field
    @property
    def DISABLE_TQDM(self) -> bool:
        return True if self.LOGGING_LEVEL == "NONE" else False

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CACHE_DIR: Path = BASE_DIR / "cache"
    PROFILING_DIR: Path = BASE_DIR / "profiling"
    DATA_DOWNLOADED_DIR: Path = BASE_DIR / "data_downloaded"
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    PRESENT_CANDIDATES_FILE: str = "present_candidates.csv"
    NUM_WORKERS: int = 8
    LOG_EVERY_N_EXAMPLES: int = 1000


settings = Settings()  # type: ignore

mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "[%(asctime)s][%(levelname)s]: %(message)s"},
        "rich": {"format": "%(message)s", "datefmt": "[%x %X]"},
    },
    "handlers": {
        "console": {
            "level": settings.LOGGING_LEVEL,
            "class": "rich.logging.RichHandler",
            "formatter": "rich",
            "markup": True,
            "show_path": True,
        },
        "file": {
            "level": settings.LOGGING_LEVEL,
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "backupCount": 1,
            "filename": "logs/app.log",
        },
    },
    "loggers": {
        "root": {
            "level": settings.LOGGING_LEVEL,
            "handlers": ["console", "file"],
        },
    },
}

if settings.LOGGING_LEVEL != "NONE":
    logging.config.dictConfig(config=LOGGING)


__all__ = ["settings"]
