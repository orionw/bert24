#!/bin/bash
PREV_JOB=$1

# Extract job details from the previous job
JOB_INFO=$(scontrol show job $PREV_JOB)
NAME=$(echo "$JOB_INFO" | grep JobName | awk -F'=' '{print $2}')
NODE=$(echo "$JOB_INFO" | grep NodeList | awk -F'=' '{print $2}')

# Function to submit next job
submit_job() {
    local dependency=$1
    local dependency_arg=""
    if [ ! -z "$dependency" ]; then
        dependency_arg="--dependency=afterany:${dependency}"
    fi
    
    sbatch --parsable \
        --job-name=${NAME} \
        --partition=h100 \
        --nodelist=${NODE} \
        --nodes=1 \
        --gpus=4 \
        --cpus-per-task=50 \
        --mem=512G \
        --output=/scratch/bvandur1/oweller2/retrieval_pretraining/logs/long-term-${NAME}/%x_%j.log \
        --error=/scratch/bvandur1/oweller2/retrieval_pretraining/logs/long-term-${NAME}/%x_%j.log \
        $dependency_arg \
        --wrap="SCRIPT_DIR=\$(dirname \$(scontrol show job \$SLURM_JOB_ID | grep Command | awk -F'=' '{print \$2}' | awk '{print \$1}')); sleep 43200; \${SCRIPT_DIR}/$(basename $0) \$SLURM_JOB_ID"
}

# Submit two new jobs: one dependent on the previous job and one dependent on the newly submitted job
new_job=$(submit_job $PREV_JOB)
submit_job $new_job

