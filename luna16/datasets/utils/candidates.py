import logging

import pandas as pd
from diskcache import FanoutCache

from luna16.settings import settings

from ... import dto
from . import utils

_log = logging.getLogger(__name__)

Diameter = float
DiameterDict = dict[str, list[tuple[dto.CoordinatesXYZ, Diameter]]]

cache = FanoutCache(
    directory=settings.CACHE_DIR / "candidates",
    timeout=1,
    size_limit=3e11,
)


def get_grouped_candidates_with_malignancy_info(
    *, require_on_disk_bool: bool = True, length: int | None = None
) -> dict[str, list[dto.CandidateMalignancyInfo]]:
    candidates: list[dto.CandidateMalignancyInfo] = (
        get_shortened_candidates_with_malignancy_info(
            require_on_disk_bool=require_on_disk_bool,
            length=length,
        )
    )
    grouped_candidates: dict[str, list[dto.CandidateMalignancyInfo]] = {}

    for candidate in candidates:
        grouped_candidates.setdefault(candidate.series_uid, []).append(candidate)

    return grouped_candidates


def get_shortened_candidates_with_malignancy_info(
    *, require_on_disk_bool: bool = True, length: int | None = None
) -> list[dto.CandidateMalignancyInfo]:
    candidates = get_candidates_with_malignancy_info(
        require_on_disk_bool=require_on_disk_bool
    )
    if length:
        return candidates[:length]
    return candidates


@cache.memoize(typed=True)
def get_candidates_with_malignancy_info(
    *, require_on_disk_bool: bool = True
) -> list[dto.CandidateMalignancyInfo]:
    """Because annotation files provided by LuNA 16 datasets were not sufficiently cleaned,
    this function uses extended and cleaned annotations with malignancy data. Because of this,
    we do not need to do the diameter gathering manually as we did in (_)get_candidates_info."""
    ct_series_uids = utils.get_series_uid_of_cts_present()

    candidates = utils.create_candidates()
    annotations = utils.create_annotations_with_malignancy()

    candidates_info: list[dto.CandidateMalignancyInfo] = []
    for _, candidate in candidates.iterrows():
        series_uid: str = str(candidate["seriesuid"])
        if require_on_disk_bool and series_uid not in ct_series_uids:
            continue
        is_nodule: bool = bool(candidate["class"])
        candidate_center = dto.CoordinatesXYZ(
            x=candidate["coordX"], y=candidate["coordY"], z=candidate["coordZ"]
        )
        if not is_nodule:
            candidates_info.append(
                dto.CandidateMalignancyInfo(
                    series_uid=series_uid,
                    center=candidate_center,
                    diameter_mm=0.0,
                    is_nodule=False,
                    is_annotated=False,
                    is_malignant=False,
                )
            )

    for _, annotation in annotations.iterrows():
        series_uid: str = str(annotation["seriesuid"])
        if require_on_disk_bool and series_uid not in ct_series_uids:
            continue
        candidate_center = dto.CoordinatesXYZ(
            x=annotation["coord_x"], y=annotation["coord_y"], z=annotation["coord_z"]
        )
        diameter_mm: float = float(annotation["diameter_mm"])
        is_malignant: bool = bool(annotation["is_malignant"])
        candidates_info.append(
            dto.CandidateMalignancyInfo(
                series_uid=series_uid,
                diameter_mm=diameter_mm,
                center=candidate_center,
                is_malignant=is_malignant,
                is_annotated=True,
                is_nodule=True,
            )
        )

    candidates_info.sort(key=lambda ci: ci.diameter_mm, reverse=True)
    return candidates_info


# @functools.lru_cache(maxsize=1)
@cache.memoize(typed=True)
def get_candidates_info(
    *, require_on_disk_bool: bool = True
) -> list[dto.CandidateInfo]:
    ct_series_uids = utils.get_series_uid_of_cts_present()

    candidates = utils.create_candidates()
    annotations = utils.create_annotations()
    diameter_dict = _get_diameters_mapped_to_seriesuid(annotations=annotations)

    candidates_info = _get_candidates_info(
        candidates=candidates,
        require_on_disk_bool=require_on_disk_bool,
        ct_series_uids=ct_series_uids,
        diameter_dict=diameter_dict,
    )
    return candidates_info


def _get_diameters_mapped_to_seriesuid(*, annotations: pd.DataFrame) -> DiameterDict:
    diameter_dict: DiameterDict = {}
    for _, annotation in annotations.iterrows():
        series_uid = annotation["seriesuid"]
        candidate_center = dto.CoordinatesXYZ(
            x=annotation["coordX"], y=annotation["coordY"], z=annotation["coordZ"]
        )
        diameter_mm = annotation["diameter_mm"]
        diameter_dict.setdefault(series_uid, []).append((candidate_center, diameter_mm))
    return diameter_dict


def _get_candidates_info(
    *,
    candidates: pd.DataFrame,
    require_on_disk_bool: bool,
    ct_series_uids: set[str],
    diameter_dict: DiameterDict,
) -> list[dto.CandidateInfo]:
    candidates_info: list[dto.CandidateInfo] = []
    for _, candidate in candidates.iterrows():
        series_uid: str = str(candidate["seriesuid"])
        if require_on_disk_bool and series_uid not in ct_series_uids:
            continue
        is_nodule: bool = bool(candidate["class"])
        candidate_center = dto.CoordinatesXYZ(
            x=candidate["coordX"], y=candidate["coordY"], z=candidate["coordZ"]
        )
        diameter_mm = 0.0
        for annotation_center, annotation_diameter in diameter_dict.get(series_uid, []):
            if diameter_mm := _get_diameter(
                nodule_diameter=annotation_diameter,
                annotation_center=annotation_center,
                candidate_center=candidate_center,
            ):
                break

        candidates_info.append(
            dto.CandidateInfo(
                series_uid=series_uid,
                is_nodule=is_nodule,
                diameter_mm=diameter_mm,
                center=candidate_center,
            )
        )
    candidates_info.sort(key=lambda ci: ci.diameter_mm, reverse=True)
    return candidates_info


def _get_diameter(
    *,
    nodule_diameter: float,
    annotation_center: dto.CoordinatesXYZ,
    candidate_center: dto.CoordinatesXYZ,
) -> float:
    nodule_radius = nodule_diameter / 2
    annotation_candidate = zip(annotation_center, candidate_center)

    for annotation_coord, candidate_coord in annotation_candidate:
        delta_mm = abs(annotation_coord - candidate_coord)
        # The 2 here is arbitrary threshold defined by us
        if delta_mm > nodule_radius / 2:
            return 0.0
    return nodule_diameter
