import wandb
import pandas as pd
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional

# Mapping of datasets to their metrics
METRIC_MAP = {
    "sst2": "metrics/glue_sst2/MulticlassAccuracy",
    "qqp": "metrics/glue_qqp/MulticlassAccuracy",
    "qnli": "metrics/glue_qnli/MulticlassAccuracy",
    "cola": "metrics/glue_cola/MulticlassMatthewsCorrCoef",
    "mnli": "metrics/glue_mnli/MulticlassAccuracy"
}

# Key hyperparameters to track
HYPERPARAMS = ['learning_rate', 'weight_decay', 'max_duration']

def process_wandb_runs(
    entity: str,
    project: str,
    encoder_size: str = "base",
    seed_filter: int = 42,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process W&B runs to find best performing configurations.
    Combines previous separate functions into one efficient process.
    
    Args:
        entity: W&B entity name
        project: W&B project name
        encoder_size: Filter for specific encoder size (e.g., "base", "mini")
        seed_filter: Only process runs with this seed value
        output_path: Optional path to save results
        
    Returns:
        Dictionary containing best runs and their metrics
    """
    api = wandb.Api()
    
    # Make single API call to get all runs
    print(f"Fetching runs for {encoder_size} encoder...")
    runs = api.runs(f"{entity}/{project}", per_page=500)
    
    # Process runs in a single pass
    best_runs = defaultdict(lambda: {"metric_value": float("-inf")})
    sweep_info = {}
    import tqdm
    for run in tqdm.tqdm(runs):
        if run.state != "finished":
            continue
            
        try:
            # Extract run information
            sweep_name = run.displayName
            run_encoder_size = sweep_name.split("-")[1]
            task_name = sweep_name.split("-")[2]
            seed = int(sweep_name.split("=")[-1]) if "seed" in sweep_name else 42
            
            # Apply filters
            if (run_encoder_size != encoder_size or 
                seed != seed_filter or 
                task_name not in METRIC_MAP):
                continue
            
            # Store sweep information if available
            if run.sweep and sweep_name not in sweep_info:
                sweep_info[sweep_name] = run.sweep.id
            
            # Get metric value
            metric_name = METRIC_MAP[task_name]
            metric_value = run.summary.get(metric_name)
            
            if metric_value is None:
                continue

            group_key = f"encoder_size={run_encoder_size} | dataset={task_name}"
            
            # Update best run if this one is better
            if metric_value > best_runs[group_key]["metric_value"]:
                hyperparams = {
                    param: run.config.get(param, None)
                    for param in HYPERPARAMS
                }
                
                best_runs[group_key] = {
                    "metric_value": metric_value,
                    "run_id": run.id,
                    "sweep_id": run.sweep.id if run.sweep else None,
                    "sweep_name": sweep_name,
                    "metric_name": metric_name,
                    "hyperparameters": hyperparams,
                }
                
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue

    # Convert to final format
    results = {
        group_key: {
            "metric_value": float(data["metric_value"]),
            "metric_name": data["metric_name"],
            "run_id": data["run_id"],
            "sweep_id": data["sweep_id"],
            "sweep_name": data["sweep_name"],
            "hyperparameters": data["hyperparameters"],
        }
        for group_key, data in best_runs.items()
        if data.get("run_id")  # Only include successfully processed runs
    }

    if output_path:
        save_results(results, output_path)
    
    return results

def save_results(results: Dict[str, Any], output_path: str) -> pd.DataFrame:
    """Save results to JSON and CSV formats."""
    # Save full results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create DataFrame with key metrics and hyperparameters
    rows = []
    for group_key, data in results.items():
        row = {
            "group": group_key,
            "metric_name": data["metric_name"],
            "metric_value": data["metric_value"],
            "run_id": data["run_id"],
            "sweep_id": data.get("sweep_id", ""),
            "sweep_name": data.get("sweep_name", ""),
        }
        # Add hyperparameters
        if "hyperparameters" in data:
            for param, value in data["hyperparameters"].items():
                row[param] = value
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    
    return df

if __name__ == "__main__":
    # Configuration
    ENTITY = "mmarone-jhu"
    PROJECT = "better_glue_sweeps"
    MODEL_SIZE = "base"
    SEED_FILTER = 42
    
    print(f"Processing runs for {MODEL_SIZE} model...")
    results = process_wandb_runs(
        entity=ENTITY,
        project=PROJECT,
        encoder_size=MODEL_SIZE,
        seed_filter=SEED_FILTER,
        output_path=f"sweep_results_{MODEL_SIZE}.json"
    )
    
    # Print summary
    df = pd.DataFrame([
        {
            "dataset": group_key.split("dataset=")[1],
            "metric_value": data["metric_value"]
        }
        for group_key, data in results.items()
    ])
    
    print("\nMetric values by dataset:")
    summary = df.groupby("dataset")["metric_value"].first().round(4)
    print(summary.to_string())
