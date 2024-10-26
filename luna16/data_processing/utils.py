from luna16 import settings


def get_series_uid_of_cts_present() -> set[str]:
    meta_header_files = settings.DATA_DOWNLOADED_DIR.glob("**/*.mhd")
    meta_header_files_on_disc = {p.stem for p in meta_header_files}
    return meta_header_files_on_disc
