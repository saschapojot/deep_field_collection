#!/bin/bash


for job_script in train_*.sh; do
    echo "Submitting job script: $job_script"
    sbatch "$job_script"
    sleep 0.1
done
