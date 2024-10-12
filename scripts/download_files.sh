#!/usr/bin/env bash

set -e

mkdir -p data_downloaded/ct_scan_subsets/

wget https://zenodo.org/records/3723295/files/subset0.zip
unzip data_downloaded/subset0.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset0.zip

wget https://zenodo.org/records/3723295/files/subset1.zip
unzip data_downloaded/subset1.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset1.zip

wget https://zenodo.org/records/3723295/files/subset2.zip
unzip data_downloaded/subset2.zip -d data_downloaded/ct_scan_subsets/
rm data_downloaded/subset2.zip
