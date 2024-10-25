#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=2GB
#PBS -l walltime=12:00:00
#PBS -l wd
#PBS -r y
#PBS -J 11-20
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -e /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_${PBS_JOBID}_${PBS_ARRAY_INDEX}.e
#PBS -o /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_${PBS_JOBID}_${PBS_ARRAY_INDEX}.o

# Load conda and activate the THOR environment
module load python3/3.10.4
conda activate THOR

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))

echo ${directories[$PBS_ARRAY_INDEX-1]}

# Activate the conda environment
/g/data/w40/esh563/miniconda/bin/conda init
/g/data/w40/esh563/miniconda/bin/conda activate THOR
python3 /home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py ${directories[$PBS_ARRAY_INDEX-1]}