import concurrent.futures
import logging
import math
from threading import Lock

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.errors import EmptyDataError
from tqdm import tqdm

from luna16 import dto, enums, settings

from . import ct, utils

_log = logging.getLogger(__name__)

lock = Lock()


class CtCutoutService:
    def __init__(self):
        self.luna_cache_dir = settings.CACHE_DIR
        self.luna_cache_dir.mkdir(exist_ok=True, parents=True)
        self.cutout_shape = dto.CoordinatesIRC(index=32, row=48, col=48)
        self.present_candidates_path = (
            self.luna_cache_dir / settings.PRESENT_CANDIDATES_FILE
        )
        self.complete_candidates_path = settings.DATA_DIR / "complete_candidates.csv"

    def create_cutouts(self, training_length: int | None = None) -> None:
        candidates_info, candidates_by_series_uid = self._get_grouped_candidates(
            training_length
        )
        for series_uid, grouped_rows in tqdm(candidates_by_series_uid):
            self._save_ct_cutouts(str(series_uid), candidates_info, grouped_rows)

    def create_cutouts_concurrent(self, training_length: int | None = None) -> None:
        candidates_info, candidates_by_series_uid = self._get_grouped_candidates(
            training_length
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.NUM_WORKERS
        ) as executor:
            features = {
                executor.submit(
                    self._save_ct_cutouts,
                    str(series_uid),
                    candidates_info,
                    grouped_rows,
                ): (series_uid, grouped_rows)
                for series_uid, grouped_rows in candidates_by_series_uid
            }
            for future in tqdm(
                concurrent.futures.as_completed(features),
                total=len(candidates_by_series_uid),
            ):
                series_uid, grouped_rows = features[future]
                try:
                    future.result()
                except Exception as exc:
                    _log.error("%r generated an exception: %s", series_uid, exc)
                else:
                    _log.debug(
                        "%s CT image was processed to create %s cutouts.",
                        series_uid,
                        len(grouped_rows),
                    )

    def _get_grouped_candidates(  # type: ignore
        self, training_length: int | None
    ) -> tuple[pd.DataFrame, DataFrameGroupBy]:  # type: ignore
        processed_candidates = self._get_present_candidates()
        # If present candidates file is already created, and it has the same training length set,
        # return only grouped candidates that were not processed (file_path is set to null).
        if processed_candidates is not None and training_length == len(
            processed_candidates
        ):
            un_processed_candidates = processed_candidates[
                processed_candidates["file_path"].isnull()
            ]
            candidates_by_series_uid = un_processed_candidates.groupby("seriesuid")
            return processed_candidates, candidates_by_series_uid

        # Otherwise, generating candidates will start from scratch by sampling random
        # training_length samples from candidates for which we have CT scans.
        candidates_info = self._get_dataframe_of_cts_present()
        if training_length:
            candidates_info = candidates_info.groupby("class").apply(
                lambda class_sub_df: class_sub_df.sample(
                    math.ceil(
                        (len(class_sub_df) / len(candidates_info)) * training_length
                    )
                )
            )
            # Because we round up with ceil during sampling, we might end up with more than training_length
            # examples. Line below ensures we end up with exactly training_length examples.
            candidates_info = candidates_info[:training_length]
            self._check_all_classes_present(candidates_info)
        candidates_info.loc[:, "file_path"] = None
        candidates_info.loc[:, "file_path"] = candidates_info.loc[
            :, "file_path"
        ].astype("string")
        candidates_by_series_uid = candidates_info.groupby("seriesuid")
        return candidates_info, candidates_by_series_uid

    def _save_ct_cutouts(
        self, series_uid: str, candidates_info: pd.DataFrame, grouped_rows: pd.DataFrame
    ) -> None:
        ct_scan = ct.Ct.read_and_create_from_image(series_uid=str(series_uid))
        for df_index in grouped_rows.index:
            center = dto.CoordinatesXYZ(
                x=candidates_info.at[df_index, "coord_x"],
                y=candidates_info.at[df_index, "coord_y"],
                z=candidates_info.at[df_index, "coord_z"],
            )
            ct_chunk, center_irc = ct_scan.get_ct_cutout_from_center(
                center=center, cutout_shape=self.cutout_shape
            )
            center_string = f"{center_irc.index}:{center_irc.row}:{center_irc.col}"
            file_name = f"{series_uid}_{center_string}.npz"
            file_path = self.luna_cache_dir / file_name

            with lock:
                _log.debug("File %s is being added to DF.", str(file_path))
                candidates_info.at[df_index, "file_path"] = str(file_path)
            _log.debug("File %s is being created.", str(file_path))
            np.savez(
                file_path,
                ct_chunk=ct_chunk,
                center_irc=center_irc.get_array(),
            )

        # Update present candidates CSV file after every new CT scan iteration
        _log.debug(
            "Present candidates are being updated after processing series uid %s.",
            series_uid,
        )
        candidates_info.to_csv(self.present_candidates_path, index=False)

    def _get_dataframe_of_cts_present(self) -> pd.DataFrame:
        ct_series_uids = utils.get_series_uid_of_cts_present()
        candidates = self._get_candidates()
        present_candidates = candidates[candidates["seriesuid"].isin(ct_series_uids)]
        return present_candidates

    def _get_candidates(self) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=self.complete_candidates_path)

    def _get_present_candidates(self) -> pd.DataFrame | None:
        try:
            return pd.read_csv(filepath_or_buffer=self.present_candidates_path)
        except (FileNotFoundError, EmptyDataError) as error:
            _log.debug(
                "File %s could not be opened because %s",
                self.present_candidates_path,
                str(error),
            )
            return None

    def _check_all_classes_present(self, candidates: pd.DataFrame) -> None:
        n_is_malignant = len(
            candidates[candidates["class"] == enums.CandidateClass.MALIGNANT]
        )
        n_is_benign = len(
            candidates[candidates["class"] == enums.CandidateClass.BENIGN]
        )
        n_not_nodule = len(
            candidates[candidates["class"] == enums.CandidateClass.NOT_NODULE]
        )
        if n_is_malignant < 2 or n_is_benign < 2 or n_not_nodule < 2:
            raise ValueError(
                "Created dataset must have at least two malignant, two benign, "
                "and two not nodule candidate. Choose bigger training length "
                "(at least 2000 examples)."
            )
