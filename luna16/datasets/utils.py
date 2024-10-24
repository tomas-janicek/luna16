import pandas as pd

from luna16 import settings


def get_present_candidates() -> pd.DataFrame:
    present_candidates_path = settings.CACHE_DIR / settings.PRESENT_CANDIDATES_FILE
    return pd.read_csv(filepath_or_buffer=present_candidates_path)
