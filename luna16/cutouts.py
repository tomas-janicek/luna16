import numpy as np
import pandas as pd
from tqdm import tqdm

from luna16 import dto
from luna16.datasets import utils as data_utils
from luna16.settings import settings


def create_cutouts(training_length: int | None = None) -> None:
    luna_cache_dir = settings.CACHE_DIR / "luna16"
    luna_cache_dir.mkdir(exist_ok=True, parents=True)

    candidates_info = _get_dataframe_of_cts_present()
    if training_length:
        candidates_info = candidates_info.sample(training_length)
    candidates_info.loc[:, "file_path"] = None
    candidates_info.loc[:, "file_path"] = candidates_info.loc[:, "file_path"].astype(
        "string"
    )

    cutout_shape = dto.CoordinatesIRC(index=32, row=48, col=48)
    candidates_by_series_uid = candidates_info.groupby("seriesuid")
    for series_uid, grouped_rows in tqdm(candidates_by_series_uid):
        ct_scan: data_utils.Ct = data_utils.Ct.read_and_create_from_image(
            series_uid=series_uid
        )
        for df_index in grouped_rows.index:
            center = dto.CoordinatesXYZ(
                x=candidates_info.at[df_index, "coord_x"],
                y=candidates_info.at[df_index, "coord_y"],
                z=candidates_info.at[df_index, "coord_z"],
            )
            ct_chunk, positive_chunk, center_irc = ct_scan.get_ct_cutout_from_center(
                center=center, cutout_shape=cutout_shape
            )
            center_string = f"{center_irc.index}:{center_irc.row}:{center_irc.col}"
            file_name = f"{series_uid}_{center_string}.npz"
            file_path = luna_cache_dir / file_name
            candidates_info.at[df_index, "file_path"] = str(file_path)

            np.savez(
                file_path,
                ct_chunk=ct_chunk,
                positive_chunk=positive_chunk,
                center_irc=center_irc.get_array(),
            )
        # Update present candidatas CSV file after every new CT scan iteration
        candidates_info.to_csv(luna_cache_dir / "present_candidates.csv", index=False)


def _get_dataframe_of_cts_present() -> pd.DataFrame:
    ct_series_uids = data_utils.get_series_uid_of_cts_present()
    candidates = _get_candidates()
    present_candidates = candidates[candidates["seriesuid"].isin(ct_series_uids)]
    return present_candidates


def _get_candidates() -> pd.DataFrame:
    complete_candidates_path = settings.DATA_DIR / "complete_candidates.csv"
    return pd.read_csv(filepath_or_buffer=complete_candidates_path)
