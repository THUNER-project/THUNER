#!/bin/bash

base_url="https://data.rda.ucar.edu/d841006/volumes/2010/20100120/nexrad_3d_v4_2_"
output_directory="./downloaded_files"

mkdir -p $output_directory

for i in $(seq -f "%02g" 18 23); do
    url="${base_url}20100120T${i}0000Z.nc"
    wget -P $output_directory $url
done