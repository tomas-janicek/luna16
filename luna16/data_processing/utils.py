from luna16 import settings


def get_series_uid_of_cts_present() -> set[str]:
    ct_scan_subsets = settings.DATA_DOWNLOADED_DIR / "ct_scan_subsets"
    meta_header_files = ct_scan_subsets.glob("subset*/*.mhd")
    meta_header_files_on_disc = {p.stem for p in meta_header_files}
    return meta_header_files_on_disc
