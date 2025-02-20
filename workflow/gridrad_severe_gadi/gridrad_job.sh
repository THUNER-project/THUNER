#!/bin/bash
year=${1:-2010}
OUTPUT_DIR="/scratch/w40/esh563/THUNER_output"
THUNER_DIR="/home/563/esh563/THUNER"

DATA_DIR="${OUTPUT_DIR}/input_data/raw/d841006/volumes/${year}"
SCRIPT_DIR="${THUNER_DIR}/workflow/gridrad_severe_gadi"
directories=$(find ${DATA_DIR} -mindepth 1 -type d -print | sort)
filepath="${SCRIPT_DIR}/${year}_directories.txt"
echo ${directories} > ${filepath}
total_diectories=$(wc -w ${filepath} | awk '{print $1}')

# Approx 4 events can be run per normalbw node, so submit in chunks of 4
for start in $(seq 0 4 ${total_diectories}); do 
    qsub -v filepath=${filepath},year=${year},start=${start} ./gridrad_PBS.sh
done