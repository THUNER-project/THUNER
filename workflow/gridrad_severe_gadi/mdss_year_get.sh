#!/bin/bash
year=${1:-2010}
LOCAL_DIR=/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes/${year}
directories=$(ls ${LOCAL_DIR})
DEST_DIR=esh563/d841006/volumes/${year}
mdss mkdir ${DEST_DIR}
for event in ${directories}; do
    qsub -v year=${year},event=${event} mdss_event_get_PBS.sh
done