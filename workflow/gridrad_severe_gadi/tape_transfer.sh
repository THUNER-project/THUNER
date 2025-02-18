#!/bin/bash
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=05:00:00,jobfs=1GB
#PBS -l storage=gdata/w40+scratch/w40+massdata/v46
#PBS -P v46

LOCAL_DIR=/scratch/w40/esh563/THUNER_output/input_data/archive/d841006/volumes/${year}
DEST_DIR=esh563/d841006/volumes/${year}
mdss put -r ${LOCAL_DIR} ${DEST_DIR}