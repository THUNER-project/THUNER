#!/bin/bash
year=${1:-2010}
LOCAL_PATH=/scratch/w40/esh563/THUNER_output/input_data/archive/d841006/volumes/${year}
directories=$(ls ${LOCAL_PATH})
TAPE_PATH=esh563/d841006/volumes/${year}
mdss mkdir ${TAPE_PATH}
LOG_DIR=/scratch/w40/esh563/THUNER_output/PBS_log/put_${year}
mkdir -p ${LOG_DIR}
cd ${LOG_DIR}
SCRIPT_DIR=/home/esh563/THUNER/workflow/gridrad_severe_gadi
for event in ${directories}; do
    qsub -v year=${year},event=${event} ${SCRIPT_DIR}/mdss_event_put_PBS.sh
done