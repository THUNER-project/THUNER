#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=28
#PBS -l mem=128GB
#PBS -l jobfs=2GB
#PBS -l walltime=3:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -o /scratch/w40/esh563/THOR_output/PBS_log/gridrad_PBS.o
#PBS -e /scratch/w40/esh563/THOR_output/PBS_log/gridrad_PBS.e

# Load gnu-parallel
module load parallel
# Load conda and activate the THOR environment
module load python3/3.10.4
conda init
conda activate THOR

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))
script="/home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py"

# In bash, the a:b syntax says slice 15 elements of the array starting from 10th element
test_directories=("${directories[@]:1:8}")

LOG_DIR="/scratch/w40/esh563/THOR_output/PBS_log"
parallel_log="${LOG_DIR}/${year}_parallel.log"

# Run multiple days concurrently with gnu-parallel
parallel -j 4 --joblog ${parallel_log} \
    "python3 ${script} {} > ${LOG_DIR}/thor_{#}.out 2> ${LOG_DIR}/thor_{#}.err" ::: "${test_directories[@]}"
