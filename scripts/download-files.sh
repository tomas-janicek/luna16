#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

data_path=$1

mkdir -p $data_path/ct_scan_subsets/

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset0.zip
unzip $data_path/subset0.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset0.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset1.zip
unzip $data_path/subset1.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset1.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset2.zip
unzip $data_path/subset2.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset2.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset3.zip
unzip $data_path/subset3.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset3.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset4.zip
unzip $data_path/subset4.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset4.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset5.zip
unzip $data_path/subset5.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset5.zip

wget -P $data_path/ https://zenodo.org/records/3723295/files/subset6.zip
unzip $data_path/subset6.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset6.zip

wget -P $data_path/ https://zenodo.org/records/4121926/files/subset7.zip
unzip $data_path/subset7.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset7.zip

wget -P $data_path/ https://zenodo.org/records/4121926/files/subset8.zip
unzip $data_path/subset8.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset8.zip

wget -P $data_path/ https://zenodo.org/records/4121926/files/subset9.zip
unzip $data_path/subset9.zip -d $data_path/ct_scan_subsets/
rm $data_path/subset9.zip
