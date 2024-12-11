#!/bin/bash
LOCAL_DIR=/home/ewan/THUNER_output
REMOTE_DIR=/scratch/w40/esh563/THUNER_output
GADI_USERNAME=esh563
REMOTE_RUNS="${GADI_USERNAME}@gadi.nci.org.au:${REMOTE_DIR}/runs/gridrad_severe"
LOCAL_RUNS="${LOCAL_DIR}/runs/gridrad_severe"
# Copy the tar files from the remote directory to the local directory
rsync -rvP "${REMOTE_RUNS}/*.tar.gz" "${LOCAL_RUNS}/"
# Get the paths to the tar files
tar_files=$(find "${LOCAL_RUNS}" -name "*.tar.gz")
for tar_file in $tar_files; do 
    # Extract the tar files
    tar -xvzf ${tar_file} -C "${LOCAL_RUNS}/"
    file=$(basename ${tar_file} .tar.gz)
    csv_file="${LOCAL_RUNS}/${file}/records/filepaths/gridrad.csv"
    cp $csv_file "${csv_file}.remote"
    # Replace the remote data directory with the local data directory in filepaths csv
    sed "s#${REMOTE_DIR}#${LOCAL_DIR}#g" "${csv_file}.remote" > $csv_file
done

