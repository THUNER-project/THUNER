#!/bin/bash
year=${1:-2010}
LOCAL_DIR=/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes/${year}
DEST_DIR=esh563/d841006/volumes/${year}
events=$(mdss ls ${DEST_DIR})
mkdir ${LOCAL_DIR}
LOG_DIR=/scratch/w40/esh563/THUNER_output/PBS_log/get_${year}
mkdir -p ${LOG_DIR}
cd ${LOG_DIR}
SCRIPT_DIR=/home/563/esh563/THUNER/workflow/gridrad_severe_gadi
for event in ${events}; do
    qsub -v year=${year},event=${event} ${SCRIPT_DIR}/mdss_event_get_PBS.sh
done