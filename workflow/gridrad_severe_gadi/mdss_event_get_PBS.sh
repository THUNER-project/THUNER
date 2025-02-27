#!/bin/bash
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=10:00:00,jobfs=1GB
#PBS -l storage=gdata/w40+scratch/w40+massdata/v46
#PBS -P v46

LOCAL_DIR=/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes/${year}/${event}
DEST_DIR=esh563/d841006/volumes/${year}/${event}
mdss get -r ${DEST_DIR} ${LOCAL_DIR}
tar -xzvf ${LOCAL_DIR}
rm ${LOCAL_DIR}
