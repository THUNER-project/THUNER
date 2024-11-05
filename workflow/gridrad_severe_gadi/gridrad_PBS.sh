#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=56
#PBS -l mem=256GB
#PBS -l jobfs=2GB
#PBS -l walltime=6:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -o /scratch/w40/esh563/THOR_output/PBS_log/gridrad_2010/gridrad_PBS.o
#PBS -e /scratch/w40/esh563/THOR_output/PBS_log/gridrad_2010/gridrad_PBS.e

# Load gnu-parallel
module load parallel
module load nco
# Load conda and activate the THOR environment
module load python3/3.10.4
conda init
conda activate THOR

# Read the directories from the file created by the gridrad_job.sh script
directories=($(cat "${filepath}"))
SCRIPT_DIR="/home/563/esh563/THOR/workflow/gridrad_severe_gadi"

# Initialize output directory
python3 ${SCRIPT_DIR}/initialize_output_directory.py
    
# # Disable HDF flock
HDF5_USE_FILE_LOCKING=FALSE

# In bash, the a:b syntax says slice 15 elements of the array starting from 10th element
test_directories=("${directories[@]}")

LOG_DIR="/scratch/w40/esh563/THOR_output/PBS_log/gridrad_${year}"
parallel_log="${LOG_DIR}/${year}_parallel.log"

# Create a node list file from $PBS_NODEFILE
uniq $PBS_NODEFILE | awk '{print $1"/4"}' > node_list.txt

# Run multiple days concurrently with gnu-parallel
# Approx 8 cores and 32GB per event
parallel --jobs 4 -sshloginfile node_list.txt --timeout 7200 --joblog ${parallel_log} \
    "python3 ${SCRIPT_DIR}/gridrad.py {} > ${LOG_DIR}/thor_{#}.out 2> ${LOG_DIR}/thor_{#}.err" ::: "${test_directories[@]}"
