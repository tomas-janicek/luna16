import pandas as pd

from luna16 import settings


def get_present_candidates() -> pd.DataFrame:
    # TODO: Move to utilities
    present_candidates_path = settings.CACHE_DIR / "luna16" / "present_candidates.csv"
    return pd.read_csv(filepath_or_buffer=present_candidates_path)
