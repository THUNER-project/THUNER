#!/bin/bash
year=${1:-2010}
OUTPUT_DIR="/scratch/w40/esh563/THOR_output"
THOR_DIR="/home/563/esh563/THOR"
# OUTPUT_DIR="/home/ewan/THOR_output"
# THOR_DIR="/home/ewan/Documents/THOR"

DATA_DIR="${OUTPUT_DIR}/input_data/raw/d841006/volumes/${YEAR}"
SCRIPT_DIR="${THOR_DIR}/workflow/gridrad_gadi"
directories=$(find ${DATA_DIR} -mindepth 1 -type d -print | sort)
filepath="${SCRIPT_DIR}/${year}_directories.txt"
echo ${directories} > ${filepath}

qsub -v filepath=${filepath} ./gridrad_PBS.sh