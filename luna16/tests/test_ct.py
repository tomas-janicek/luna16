from luna16 import data_processing


def test_loading_ct_from_series_uid():
    series_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260"
    ct_scan = data_processing.Ct.read_and_create_from_image(series_uid=series_uid)
    assert ct_scan.series_uid == series_uid
