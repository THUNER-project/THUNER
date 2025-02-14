#!/bin/bash
#PBS -q normalbw
#PBS -l ncpus=16,mem=64GB,walltime=10:00:00,jobfs=10GB
#PBS -l storage=gdata/w40+scratch/w40
#PBS -P v46

DIR=/scratch/w40/esh563/THUNER_output/input_data/raw/d841006/volumes
days=$(ls $DIR/$year)

create_tar() {
    day=$1
    tar -czvf ${DIR}/${year}/${day}.tar.gz -C $DIR/${year} ${DIR}/${year}/${day}
}

echo $days | parallel -j 32 create_tar {}
