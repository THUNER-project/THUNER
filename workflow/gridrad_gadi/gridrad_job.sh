#!/bin/bash

DATA_DIR="/scratch/w40/esh563/THOR_output/input_data/raw/d841006/volumes/2010"
SCRIPT_DIR="/home/563/esh563/THOR/workflow/gridrad_gadi"
# DATA_DIR="/home/ewan/THOR_output/input_data/raw/d841006/volumes/2010"
# SCRIPT_DIR="/home/ewan/Documents/THOR/workflow/gridrad_gadi"
datetime=$(date +"%Y%m%d_%H%M%S")
directories=$(find ${DATA_DIR} -mindepth 1 -type d -print | sort)
directory_count=$(echo "${directories}" | wc -l)

filepath="${SCRIPT_DIR}/directories.txt"
echo ${directories} > ${filepath}

qsub -v datetime="${datetime}",filepath="${filepath}",directory_count="${directory_count}" ./gridrad_PBS.sh

