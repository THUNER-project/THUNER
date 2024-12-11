#!/bin/bash

# This script will create a custom conda installation in the users /g/data directory
export CONDA_BASE=http://repo.continuum.io/miniconda/Miniconda3
wget ${CONDA_BASE}-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "/g/data/$PROJECT/$USER/miniconda"
export PATH="/g/data/$PROJECT/$USER/miniconda/bin:$PATH"
hash -r
# Specify the full path to the conda executable to avoid conflicts with the CMS conda
/g/data/${PROJECT}/${USER}/miniconda/bin/conda init
source ~/.bashrc
/g/data/${PROJECT}/${USER}/miniconda/bin/conda update conda
# One can now use this custom conda installation as normal, e.g. to create the THUNER environment
# For now specify the full path to the custom conda executable whevenever you call 
# a "conda" command. Not sure how to make the gadi shell point to the custom conda by
# default.


