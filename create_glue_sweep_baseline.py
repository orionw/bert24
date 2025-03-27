import wandb
import yaml
import json
import copy
import os
from pathlib import Path

"""
This script creates sweeps in W&B UI for baseline model configs on GLUE tasks.
The sweep IDs are written to a JSON file which can be used to launch the sweeps.
"""

seed = 42

# Define the metric mapping for each GLUE task
metric_map = {
    "mnli": "MulticlassAccuracy",
    "cola": "MulticlassMatthewsCorrCoef",
    "qnli": "MulticlassAccuracy",
    "qqp": "MulticlassAccuracy",
    "sst2": "MulticlassAccuracy",
    "mrpc": "MulticlassAccuracy",
    "stsb": "SpearmanCorrCoef",
    "rte": "MulticlassAccuracy",
}

# Define baseline models to run sweeps for
baseline_models = [
    "microsoft/MiniLM-L12-H384-uncased",
    "huawei-noah/TinyBERT_General_4L_312D",
    "google/bert_uncased_L-4_H-256_A-4", 
    "google/bert_uncased_L-4_H-512_A-8",
    "distilbert/distilbert-base-uncased",
    "distilbert/distilroberta-base", 
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge"
]

def create_sweep_for_baseline(config_path, task, model_name):
    """
    Creates a W&B sweep for a baseline model on a specific GLUE task.
    
    Args:
        config_path (str): Path to the baseline model config file
        task (str): GLUE task name
        model_name (str): Name of the pretrained model
    
    Returns:
        str: Sweep ID
    """
    parent_task = "glue"
    print(f"Creating sweep for model {model_name} on task {task}")
    
    # Load YAML config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Define sweep parameters based on task
    epochs = [1, 2, 3, 4] if task in ["mnli", "sst2", "rte"] else [2, 5, 10, 12]
    learning_rates = [1e-5, 3e-5, 5e-5, 8e-5, 1e-4]
    weight_decays = [1e-6, 5e-6, 8e-6, 1e-5]
    batch_sizes = [16, 32]
    
    # Use different parameters for larger models to avoid OOM errors
    if "xlarge" in model_name or "xxlarge" in model_name:
        batch_sizes = [8, 16]
    
    # Create folder to store parameter-specific configs
    model_short_name = model_name.replace('/', '--')
    config_dir = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/sweep_param_configs/baseline/{task}/{model_short_name}"
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate all parameter combinations with explicit paths for sweep command
    param_configs = []
    
    for lr in learning_rates:
        for wd in weight_decays:
            for bs in batch_sizes:
                for ep in epochs:
                    # Create a unique identifier for this parameter combination
                    model_id = model_name.split('/')[-1]
                    param_str = f"{task}_{model_id}_lr{lr}_wd{wd}_bs{bs}_ep{ep}"
                    
                    # Create a copy of the config
                    param_config = copy.deepcopy(config)
                    
                    # Update with the parameter values
                    save_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/ft_sweeps/baseline/{task}/{model_short_name}/{param_str}"
                    param_config["save_finetune_checkpoint_prefix"] = save_path
                    param_config["save_finetune_checkpoint_folder"] = f"{save_path}_folder"
                    param_config["base_run_name"] = param_str
                    param_config["learning_rate"] = lr
                    param_config["weight_decay"] = wd
                    param_config["device_train_microbatch_size"] = bs
                    param_config["max_duration"] = ep
                    
                    # Update model configuration
                    param_config["model"]["pretrained_model_name"] = model_name
                    param_config["tokenizer_name"] = model_name
                    param_config["model"]["tokenizer_name"] = model_name
                    
                    # Save to a new config file
                    param_config_path = f"{config_dir}/{param_str}.yaml"
                    with open(param_config_path, 'w') as f:
                        yaml.dump(param_config, f)
                    
                    # Add to list of config paths for parameters
                    param_configs.append({"path": param_config_path, "name": param_str})
    
    # Create sweep with the generated configs
    sweep_config = {
        "name": f"baseline-{model_short_name}-{task}-{seed}",
        "command": [
            "${interpreter}",
            "${program}",
            "${args}"
        ],
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": f"metrics/{parent_task}_{task}/{metric_map[task]}"
        },
        "parameters": {
            "config_path": {"values": [p["path"] for p in param_configs]},
        },
        "program": "eval.py",
    }

    sweep_id = wandb.sweep(sweep_config, project="baseline_glue_sweeps", entity="mmarone-jhu")
    print(f"Created sweep {sweep_id} for model {model_name} on task {task}")
    return sweep_id

def create_baseline_config(task_config_path, model_name):
    """
    Creates a new config file for a baseline model by updating the task config.
    
    Args:
        task_config_path (str): Path to the task-specific config file
        model_name (str): Name of the pretrained model
    
    Returns:
        str: Path to the newly created config file for the sweep
    """
    # Extract task name from path
    task = task_config_path.split('/')[-1].replace(".yaml", "")
    
    # Load the task-specific config
    try:
        with open(task_config_path, 'r') as f:
            new_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Task config file not found at {task_config_path}")
        return None
    
    # Update model-specific configuration
    new_config["model"]["pretrained_model_name"] = model_name
    new_config["tokenizer_name"] = model_name
    new_config["model"]["tokenizer_name"] = model_name
    
    # Set run name
    model_short_name = model_name.replace("/", "--")
    new_config["base_run_name"] = f"{model_short_name}-{task}-seed-{seed}"
    new_config["default_seed"] = seed
    
    # Update save paths
    new_config["save_finetune_checkpoint_prefix"] = f"./ft_sweeps/baseline/{task}/{model_short_name}"
    new_config["save_finetune_checkpoint_folder"] = f"{new_config['save_finetune_checkpoint_prefix']}_folder"
    
    # Create output path for the new config
    output_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/baseline/{model_short_name}-{task}.yaml"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the new config
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    print(f"Created new config at {output_path}")
    
    return output_path

# Define GLUE tasks to run
glue_tasks = [
    "mnli",
    # "cola",
    "qnli",
    "qqp",
    "sst2",
    # Stage 2 tasks - uncomment to run these as well
    # "mrpc",
    # "stsb",
    # "rte",
]

def main():
    # Main execution
    sweep_id_map = {}
    
    for task in glue_tasks:
        task_config_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/baseline/tasks/{task}.yaml"
        
        # Check if task config exists
        if not os.path.exists(task_config_path):
            print(f"Warning: Task config not found for {task} at {task_config_path}, skipping.")
            continue
        
        for model_name in baseline_models:
            try:
                # Create the baseline config
                config_path_for_sweep = create_baseline_config(task_config_path, model_name)
                if not config_path_for_sweep:
                    continue
                
                # Create the sweep
                sweep_id = create_sweep_for_baseline(config_path_for_sweep, task, model_name)
                
                # Store the sweep ID
                sweep_key = f"baseline-{model_name.replace('/', '--')}-{task}-{seed}"
                sweep_id_map[sweep_key] = sweep_id
            except Exception as e:
                print(f"Error creating sweep for {model_name} on {task}: {e}")
                continue

    # Save sweep IDs to a JSON file
    with open('baseline_sweep_ids.json', 'w') as f:
        json.dump(sweep_id_map, f, indent=4)

    print(f"Created {len(sweep_id_map)} sweeps. Sweep IDs saved to baseline_sweep_ids.json")

if __name__ == "__main__":
    main()