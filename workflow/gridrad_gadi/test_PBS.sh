#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=2GB
#PBS -l walltime=3:00:00
#PBS -l wd
#PBS -r y
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40

# Load conda and activate the THOR environment
module load python3/3.10.4
conda init
conda activate THOR

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))
# Run the gridrad.py script with the directory corresponding to the current PBS_ARRAY_INDEX
python3 /home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py /scratch/w40/esh563/THOR_output/input_data/raw/d841006/volumes/2010/20100515
