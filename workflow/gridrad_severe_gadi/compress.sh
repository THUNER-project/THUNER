#!/bin/bash
#PBS -q normalbw
#PBS -l ncpus=32,mem=128GB,walltime=02:00:00,jobfs=10GB
#PBS -l storage=gdata/w40+scratch/w40
#PBS -P v46

module load parallel
DATA_DIR="/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes"

create_tar() {
    tar -czvf ${1}.tar.gz -C $(dirname ${1}) $(basename ${1})
}
export -f create_tar

find ${DATA_DIR}/${year} -mindepth 1 -maxdepth 1 -type d | parallel -j 32 create_tar
