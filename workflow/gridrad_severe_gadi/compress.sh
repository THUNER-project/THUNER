#!/bin/bash
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=01:00:00,jobfs=10GB
#PBS -l storage=gdata/w40+scratch/w40
#PBS -P v46

DIR=/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes
for year in {2010..2019}; do
    tar -czvf ${DIR}/${year}.tar.gz -C $DIR ${DIR}/${year}
done
