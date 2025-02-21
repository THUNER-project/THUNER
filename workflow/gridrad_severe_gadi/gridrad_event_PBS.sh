#!/bin/bash
#PBS -q normalbw
#PBS -P v46
#PBS -l ncpus=4
#PBS -l mem=32GB
#PBS -l jobfs=2GB
#PBS -l walltime=6:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -o /scratch/w40/esh563/THUNER_output/PBS_log/gridrad_2010/gridrad_PBS.o
#PBS -e /scratch/w40/esh563/THUNER_output/PBS_log/gridrad_2010/gridrad_PBS.e

# Load gnu-parallel
module load parallel
module load nco
# Load conda and activate the THUNER environment
module load python3/3.10.4
conda init
conda activate THUNER

# Read the directories from the file created by the gridrad_job.sh script
SCRIPT_DIR="/home/563/esh563/THUNER/workflow/gridrad_severe_gadi"

# Initialize output directory
python3 ${SCRIPT_DIR}/initialize_output_directory.py
    
# # Disable HDF flock
HDF5_USE_FILE_LOCKING=FALSE

python3 ${SCRIPT_DIR}/gridrad.py ${event_dir}