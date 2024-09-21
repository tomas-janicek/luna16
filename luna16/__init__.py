import logging.config

from luna16 import settings

logging.config.dictConfig(config=settings.LOGGING)
