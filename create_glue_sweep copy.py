import wandb
import yaml
import json
import copy
"""
This script creates sweeps in the UI from a list of configs. The sweep IDs are written to a text file.
Those sweeps are then run with launch_sweeps.sh script.
"""

seed = 42


metric_map = {
    "mnli": "MulticlassAccuracy",
    "cola": "MulticlassMatthewsCorrCoef",
    "qnli": "MulticlassAccuracy",
    "qqp": "MulticlassAccuracy",
    "sst2": "MulticlassAccuracy",
    "mrpc": "MulticlassAccuracy",
    "stsb": "SpearmanCorrCoef",
}

def create_sweep_for_config(config_path, task, model_type, model_size):
    parent_task = "glue"
    print(f"found task {task} of parent task : {parent_task}")
    
    # load yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    epochs = [1, 2, 3, 4] if task in ["mnli", "sst2", "rte"] else [2, 5, 10, 12]

    # Define sweep parameters
    learning_rates = [1e-5, 3e-5, 5e-5, 8e-5, 1e-4]
    weight_decays = [1e-6, 5e-6, 8e-6, 1e-5]
    batch_sizes = [16, 32]
    
    # Create folder to store parameter-specific configs
    import os
    config_dir = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/sweep_param_configs/{task}/{model_type}/{model_size}"
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate all parameter combinations with explicit paths for sweep command
    param_configs = []
    config_commands = []
    
    for lr in learning_rates:
        for wd in weight_decays:
            for bs in batch_sizes:
                for ep in epochs:
                    # Create a unique identifier for this parameter combination
                    param_str = f"{task}_{model_type}_{model_size}_lr{lr}_wd{wd}_bs{bs}_ep{ep}"
                    
                    # Create a copy of the config
                    param_config = copy.deepcopy(config)
                    
                    # Update with the parameter values
                    param_config["local_pretrain_checkpoint_folder"] = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/ft_sweeps/{task}/{model_type}/{model_size}/{param_str}-pretrain"
                    param_config["save_finetune_checkpoint_prefix"] = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/ft_sweeps/{task}/{model_type}/{model_size}/{param_str}"
                    param_config["base_run_name"] = param_str
                    param_config["learning_rate"] = lr
                    param_config["weight_decay"] = wd
                    param_config["device_train_microbatch_size"] = bs
                    param_config["max_duration"] = ep
                    
                    # Save to a new config file
                    param_config_path = f"{config_dir}/{param_str}.yaml"
                    with open(param_config_path, 'w') as f:
                        yaml.dump(param_config, f)
                    
                    # Add to list of config paths for parameters
                    param_configs.append({"path": param_config_path, "name": param_str})
    
    # Create sweep with the generated configs
    # Create a command for each config directly
    sweep_config = {
        "command": [
            "python",
            "eval.py",
            "${config_path}"
        ],
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": f"metrics/{parent_task}_{task}/{metric_map[task]}"
        },
        "parameters": {
            "config_path": {"values": [p["path"] for p in param_configs]},
            "config_name": {"values": [p["name"] for p in param_configs]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="encoder_glue_sweeps_fixed_save", entity="mmarone-jhu")
    return sweep_id

def create_new_config_for_sweep(task_config_path, model_size, model_type, best_checkpoint_map=None):
    """
    Creates a new config file for a sweep by updating task config with model-specific settings.
    
    Args:
        task_config_path (str): Path to the task-specific config file
        model_size (str): Size of the model (very_tiny, tiny, small, base, large, huge)
        model_type (str): Type of model (encoder or decoder)
        best_checkpoint_map (str, optional): Path to JSON file mapping tasks to best checkpoints for stage 2
    
    Returns:
        str: Path to the newly created config file for the sweep
    """
    # Load the size-specific model config
    size_config_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/retrieval_pretraining/scripts/{model_size}_config.json"
    with open(size_config_path, 'r') as f:
        size_config = json.load(f)
    
    # Load the task-specific config
    with open(task_config_path, 'r') as f:
        new_config = yaml.safe_load(f)
    
    # Update only the specific model architecture fields
    new_config["model"]["model_config"].update({
        "hidden_size": size_config["hidden_size"],
        "intermediate_size": size_config["intermediate_size"],
        "num_attention_heads": size_config["num_attention_heads"],
        "num_hidden_layers": size_config["num_hidden_layers"]
    })
    
    # Set model type specific configuration
    if model_type == "decoder":
        new_config["model"]["model_config"].update({
            "causal_mask": True,
            "masked_prediction": False,
            "pad_logits": True
        })
    else:  # encoder
        new_config["model"]["model_config"].update({
            "causal_mask": False,
            "masked_prediction": True,
            "pad_logits": False
        })
    
    # Define the checkpoint mapping
    model_dirs = {
        "encoder": {
            "very_tiny": "encoder_very_tiny_no_packing_prolong_decay_lower_mask",
            "tiny": "encoder_tiny_no_packing_v2_prolong_decay_lower_mask",
            "mini": "encoder_mini_no_packing_prolong_decay_lower_mask",
            "base": "encoder_base_no_packing_prolong_decay_lower_mask",
            "large": "encoder_large_no_packing_prolong_decay_lower_mask",
            "huge": "encoder_huge_no_packing_prolong_decay_lower_mask"
        },
        "decoder": {
            "very_tiny": "decoder_very_tiny_no_packing_prolong_decay",
            "tiny": "decoder_tiny_no_packing_v2_prolong_decay",
            "mini": "decoder_mini_no_packing_prolong_decay",
            "base": "decoder_base_no_packing_v3_prolong_decay",
            "large": "decoder_large_no_packing_prolong_decay",
            "huge": "decoder_huge_no_packing_prolong_decay"
        }
    }

    # Get the task name
    task = task_config_path.split('/')[-1].replace(".yaml", "").replace("-sweep", "")
    model_name = f"{model_type}-{model_size}-{task}"
    new_config["base_run_name"] = model_name + f"-seed-{seed}"
    new_config["default_seed"] = seed
    new_config["tasks"]["abc"]["seeds"] = [seed]


    new_config["local_pretrain_checkpoint_folder"] = f"./ft_sweeps/{model_name}-pretrain-{model_type}-{task}"
    new_config["save_finetune_checkpoint_prefix"] = f"./ft_sweeps/{model_name}-{model_type}-{task}"

    new_config["device_train_microbatch_size"] = 64 # will get overwritten by sweep
    
    # Handle checkpoints
    if best_checkpoint_map and task in ["mrpc", "stsb", "rte"]:
        # Use best checkpoint for stage 2 tasks
        with open(best_checkpoint_map, 'r') as f:
            checkpoint_mapping = json.load(f)
        
        checkpoint_key = f"{model_type}-{model_size}-{task}"
        if checkpoint_key in checkpoint_mapping:
            new_config["starting_cp"] = checkpoint_mapping[checkpoint_key]
    else:
        # Use default checkpoint from model_dirs mapping
        model_dir = model_dirs[model_type][model_size]
        new_config["starting_cp"] = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/models_pythia_like/{model_dir}/latest-rank0.pt"
    
    # Create output path for the new config
    output_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/{model_type}-{model_size}-{task}.yaml"
    
    # Save the new config
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    print(f"created new config at {output_path}")
    
    return output_path


# configs to run
stage1_configs = [
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/mnli.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/cola.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/qnli.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/qqp.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/sst2.yaml",

]

sweep_id_map = {}
for config_path in stage1_configs:
    for model_size in ["very_tiny", "tiny", "mini", "base", "huge"]: # "large"
        for model_type in ["encoder"]: # , "decoder"]:
            # create the config for this yaml file and save it to the same path
            config_path_for_sweep = create_new_config_for_sweep(config_path, model_size, model_type)
            task = config_path.split('/')[-1].replace(".yaml", "").replace("-sweep", "")
            sweep_id = create_sweep_for_config(config_path_for_sweep, task, model_size, model_type)
            sweep_id_map[f"{model_type}-{model_size}-{task}-{seed}"] = sweep_id

# save sweep IDs to a txt
with open('sweep_ids.json', 'w') as f:
    json.dump(sweep_id_map, f, indent=4)

# stage2_configs = [
#     "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/mrpc.yaml",
#     "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/stsb.yaml",
#     "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/rte.yaml",
# ]

# best_checkpoint_map = "best_checkpoint_map.json"
# all_sweep_ids = {}
# for config_path in stage2_configs:
#     for model_size in ["very_tiny", "tiny", "small", "base", "large", "huge"]:
#         for model_type in ["encoder", "decoder"]:
#             # create the config for this yaml file and save it to the same path
#             config_path_for_sweep = create_new_config_for_sweep(config_path, model_size, model_type, best_checkpoint_map)
#             # sweep_id = create_sweep_for_config(config_path_for_sweep)
#             # all_sweep_ids[config_path_for_sweep] = sweep_id

# with open('sweep_ids_stage2.txt', 'w') as f:
#     for config_path, sweep_id in all_sweep_ids.items():
#         f.write(f"{sweep_id},{config_path}\n")


# then run wanbd agents for slurm
