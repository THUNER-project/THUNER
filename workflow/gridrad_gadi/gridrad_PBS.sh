#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=10GB
#PBS -l walltime=6:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -J 40-50
#PBS -e /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_job.e
#PBS -o /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_job.o

# This job should be submitted using the gridrad_job.sh script, which sets the 
# variables datetime, filepath and directory_count.

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))

/g/data/w40/esh563/miniconda/bin/conda init
/g/data/w40/esh563/miniconda/bin/conda activate THOR
script_path="/home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py"
python3 /home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py ${directories[${PBS_ARRAY_INDEX}-1]}