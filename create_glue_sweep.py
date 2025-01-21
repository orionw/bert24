import wandb
import yaml
import json

"""
This script creates sweeps in the UI from a list of configs. The sweep IDs are written to a text file.
Those sweeps are then run with launch_sweeps.sh script.
"""


def create_sweep_for_config(config_path, task):
    parent_task = "glue"
    print(f"found task {task} of parent task : {parent_task}")
    
    # load yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    sweep_config = {
        "name": config_path.split('/')[-1].replace("-sweep", "").replace(".yaml", ""),
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            config_path,
            "${args}"
        ],
        "method": "random",
        "metric": {
            "goal": "maximize",
            "name": f"metrics/{parent_task}_{task}/MulticlassAccuracy"
        },
        "parameters": {
            "device_train_microbatch_size": {"values": [64]},
            "task": {"values": [task]},
            "starting_cp": {"values": [config["starting_cp"]]},
            "learning_rate": {"values": [1e-5, 3e-5, 5e-5, 8e-5]},
            "max_duration": {"values": [1, 2, 3]},
        },
        "program": "eval.py",
        "run_cap": 60,
    }

    sweep_id = wandb.sweep(sweep_config, project="better_glue_sweeps", entity="mmarone-jhu")


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
            "very_tiny": "encoder_very_tiny_no_packing",
            "tiny": "encoder_tiny_no_packing_v2",
            "mini": "encoder_mini_no_packing",
            "base": "encoder_no_packing",
            "large": "encoder_large_no_packing",
            "huge": "encoder_huge_no_packing"
        },
        "decoder": {
            "very_tiny": "decoder_very_tiny_no_packing",
            "tiny": "decoder_tiny_no_packing_v2",
            "mini": "decoder_mini_no_packing",
            "base": "decoder_base_no_packing_v3",
            "large": "decoder_large_no_packing_v2",
            "huge": "decoder_huge_no_packing"
        }
    }
    
    # Get the task name
    task = task_config_path.split('/')[-1].replace(".yaml", "").replace("-sweep", "")
    model_name = f"{model_type}-{model_size}-{task}"
    new_config["base_run_name"] = model_name

    # update the save folders, e.g. local_pretrain_checkpoint_folder and save_finetune_checkpoint_prefix
    # TODO
    new_config["local_pretrain_checkpoint_folder"] = f"./bert-finetune-checkpoints-best-{model_name}-pretrain-{model_type}"
    new_config["save_finetune_checkpoint_prefix"] = f"./bert-finetune-checkpoints-best-{model_name}-{model_type}"
    
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
        new_config["starting_cp"] = f"latest-rank0/{model_dir}"
    
    # Create output path for the new config
    output_path = f"/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/{model_type}-{model_size}-{task}.yaml"
    
    breakpoint()
    # Save the new config
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    return output_path


# configs to run
stage1_configs = [
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/mnli.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/cola.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/qnli.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/qqp.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/sst2.yaml",

]

all_sweep_ids = {}
for config_path in stage1_configs:
    for model_size in ["very_tiny", "tiny", "mini", "base", "large", "huge"]:
        for model_type in ["encoder", "decoder"]:
            # create the config for this yaml file and save it to the same path
            config_path_for_sweep = create_new_config_for_sweep(config_path, model_size, model_type)
            task = config_path.split('/')[-1].replace(".yaml", "").replace("-sweep", "")
            # sweep_id = create_sweep_for_config(config_path_for_sweep, task)
            # all_sweep_ids[config_path_for_sweep] = sweep_id

# # save sweep IDs to a txt
# with open('sweep_ids.txt', 'w') as f:
#     for config_path, sweep_id in all_sweep_ids.items():
#         f.write(f"{sweep_id},{config_path}\n")

stage2_configs = [
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/mrpc.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/stsb.yaml",
    "/home/oweller2/my_scratch/retrieval_pretraining/bert24/yamls/finetuning/glue/tasks/rte.yaml",
]

best_checkpoint_map = "best_checkpoint_map.json"
all_sweep_ids = {}
for config_path in stage2_configs:
    for model_size in ["very_tiny", "tiny", "small", "base", "large", "huge"]:
        for model_type in ["encoder", "decoder"]:
            # create the config for this yaml file and save it to the same path
            config_path_for_sweep = create_new_config_for_sweep(config_path, model_size, model_type, best_checkpoint_map)
            # sweep_id = create_sweep_for_config(config_path_for_sweep)
            # all_sweep_ids[config_path_for_sweep] = sweep_id

# with open('sweep_ids_stage2.txt', 'w') as f:
#     for config_path, sweep_id in all_sweep_ids.items():
#         f.write(f"{sweep_id},{config_path}\n")
