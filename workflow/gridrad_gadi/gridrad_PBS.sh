#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l jobfs=2GB
#PBS -l walltime=3:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40

# Load gnu-parallel
module load parallel
# Load conda and activate the THOR environment
module load python3/3.10.4
conda init
conda activate THOR

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))
script="/home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py"

# For testing: Use only the first 5 entries of directories
test_directories=("${directories[@]:10:15}")

# Run the gridrad.py script with the directory corresponding to the current PBS_ARRAY_INDEX
parallel -j 2 python3 $script ::: "${test_directories[@]}"
