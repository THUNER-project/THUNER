#!/bin/bash
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=10:00:00,jobfs=1GB
#PBS -l storage=gdata/w40+scratch/w40+massdata/v46
#PBS -P v46

LOCAL_PARENT=/scratch/w40/esh563/THUNER_output/input_data/archive/d841006/volumes
LOCAL_PATH=${LOCAL_PARENT}/${year}/${event}
TAPE_PATH=esh563/d841006/volumes/${year}/${event}
mdss put -r ${LOCAL_PATH} ${TAPE_PATH}
