#!/usr/bin/env bash

set -e

mkdir -p data_downloaded/ct_scan_subsets/

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset0.zip
unzip data_downloaded/subset0.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset0.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset1.zip
unzip data_downloaded/subset1.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset1.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset2.zip
unzip data_downloaded/subset2.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset2.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset3.zip
unzip data_downloaded/subset3.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset3.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset4.zip
unzip data_downloaded/subset4.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset4.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset5.zip
unzip data_downloaded/subset5.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset5.zip

wget -P data_downloaded/ https://zenodo.org/records/3723295/files/subset6.zip
unzip data_downloaded/subset6.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset6.zip

wget -P data_downloaded/ https://zenodo.org/records/4121926/files/subset7.zip
unzip data_downloaded/subset7.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset7.zip

wget -P data_downloaded/ https://zenodo.org/records/4121926/files/subset8.zip
unzip data_downloaded/subset8.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset8.zip

wget -P data_downloaded/ https://zenodo.org/records/4121926/files/subset9.zip
unzip data_downloaded/subset9.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset9.zip
