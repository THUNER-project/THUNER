#!/bin/bash
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=10:00:00
#PBS -l storage=gdata/w40+scratch/w40
#PBS -P w40
/g/data/w40/esh563/globus/globusconnectpersonal-3.2.5/globusconnectpersonal -start
