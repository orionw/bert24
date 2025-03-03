#!/bin/bash

# Check if a file path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <json_file_path>"
    exit 1
fi

JSON_FILE=$1

# Check if the file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File '$JSON_FILE' does not exist."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: This script requires 'jq'. Please install it first."
    echo "You can install it with: sudo apt-get install jq (Debian/Ubuntu) or brew install jq (macOS)"
    exit 1
fi

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "Error: This script requires 'wandb'. Please install it first."
    echo "You can install it with: pip install wandb"
    exit 1
fi

# Set the entity and project name
# You may want to modify these or pass them as arguments
ENTITY="mmarone-jhu"
PROJECT="better_glue_sweeps"

# Function to run wandb agent for a sweep
run_sweep() {
    local model_name=$1
    local sweep_id=$2
    
    echo "Running wandb agent for $model_name with sweep ID: $sweep_id"
    echo "Command: wandb agent --entity $ENTITY --project $PROJECT $sweep_id"
    
    # Run the wandb agent command
    wandb agent --entity "$ENTITY" --project "$PROJECT" "$sweep_id"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: wandb agent for $model_name failed."
    else
        echo "wandb agent for $model_name completed successfully."
    fi
    
    echo "----------------------------------------"
}

# Process each key-value pair in the JSON file
jq -r 'to_entries[] | "\(.key) \(.value)"' "$JSON_FILE" | while read -r model_name sweep_id; do
    run_sweep "$model_name" "$sweep_id"
done

echo "All sweeps completed."