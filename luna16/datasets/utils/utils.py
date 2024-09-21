import functools
import logging

import pandas as pd

from luna16.settings import settings

from ... import dto

_log = logging.getLogger(__name__)

Diameter = float
DiameterDict = dict[str, list[tuple[dto.CoordinatesXYZ, Diameter]]]


@functools.lru_cache(maxsize=1)
def get_series_uid_of_cts_present() -> set[str]:
    ct_scan_subsets = settings.DATA_DOWNLOADED_DIR / "ct_scan_subsets"
    meta_header_files = ct_scan_subsets.glob("subset*/*.mhd")
    meta_header_files_on_disc = {p.stem for p in meta_header_files}
    return meta_header_files_on_disc


@functools.lru_cache(maxsize=1)
def create_candidates() -> pd.DataFrame:
    candidates_path = settings.LUNA_16_DATA_DIR / "candidates.csv"
    return pd.read_csv(filepath_or_buffer=candidates_path)


@functools.lru_cache(maxsize=1)
def create_annotations() -> pd.DataFrame:
    annotations_path = settings.LUNA_16_DATA_DIR / "annotations.csv"
    return pd.read_csv(filepath_or_buffer=annotations_path)


@functools.lru_cache(maxsize=1)
def create_annotations_with_malignancy() -> pd.DataFrame:
    annotations_path = settings.LUNA_16_DATA_DIR / "annotations_with_malignancy.csv"
    return pd.read_csv(filepath_or_buffer=annotations_path)
