#!/bin/bash
#PBS -q normalbw
#PBS -P w40
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=10GB
#PBS -l walltime=6:00:00
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/w40+gdata/rq0+scratch/w40
#PBS -e /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_job_$DATETIME.e
#PBS -o /home/563/esh563/THOR/workflow/gridrad_gadi/PBS_jobs/gridrad_job_$DATETIME.o

/g/data/w40/esh563/miniconda/bin/conda init
/g/data/w40/esh563/miniconda/bin/conda activate THOR
python3 /home/563/esh563/THOR/workflow/gridrad_gadi/gridrad.py
