#!/usr/bin/env python
import json
import os
import time
from collections import defaultdict
import argparse
import wandb

def extract_and_delete_failed_runs(results_file="encoder_sweep_results.json", 
                           output_file="failed_run_urls.txt",
                           entity="mmarone-jhu", 
                           project="encoder_glue_sweeps_fixed_save",
                           tasks=None,
                           model_sizes=None,
                           delete=False,
                           dry_run=True,
                           delay=0.5):
    """
    Extract URLs for all failed runs from the results file and optionally delete them
    """
    # Load the results file
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found. Run the main script first.")
        return
        
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    failed_urls = []
    failed_runs = []
    
    print("\n=== Extracting URLs for Failed Runs ===")
    
    # Create a dictionary to group failures by task
    task_failures = defaultdict(list)
    
    for config_name, result in results.items():
        # Apply filters if specified
        if tasks and result.get("task") not in tasks:
            continue
        if model_sizes and result.get("model_size") not in model_sizes:
            continue
            
        task = result.get("task")
        model_size = result.get("model_size")
        sweep_id = result.get("sweep_id")
        
        # Skip if missing key data
        if not all([task, model_size, sweep_id]):
            continue
            
        runs = result.get("runs", [])
        
        for run in runs:
            if run.get("is_failed", False):
                run_id = run.get("id")
                run_name = run.get("name", "unnamed")
                state = run.get("state", "unknown")
                reason = run.get("failure_reason") or "Unknown"
                
                # Construct the URL to the run
                run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                
                # Add to the list
                failed_urls.append(run_url)
                failed_runs.append({
                    "id": run_id,
                    "url": run_url,
                    "config": config_name,
                    "model_size": model_size,
                    "task": task,
                    "run_name": run_name,
                    "state": state,
                    "reason": reason
                })
                
                # Add to task-specific list
                task_failures[task].append({
                    "id": run_id,
                    "url": run_url,
                    "config": config_name,
                    "model_size": model_size,
                    "run_name": run_name,
                    "state": state,
                    "reason": reason
                })
    
    # Write URLs to file
    with open(output_file, 'w') as f:
        for url in failed_urls:
            f.write(f"{url}\n")
    
    # Print summary by task
    total_failures = len(failed_urls)
    print(f"Found {total_failures} failed runs across all tasks/models")
    
    for task, failures in sorted(task_failures.items()):
        print(f"\n=== Task: {task} ({len(failures)} failed runs) ===")
        
        # Group by model size
        size_groups = defaultdict(list)
        for failure in failures:
            size_groups[failure["model_size"]].append(failure)
        
        for size, size_failures in sorted(size_groups.items()):
            print(f"  Model Size: {size} ({len(size_failures)} failed runs)")
            
            # Group by failure reason
            reason_groups = defaultdict(list)
            for failure in size_failures:
                reason_groups[failure["reason"]].append(failure)
            
            for reason, reason_failures in sorted(reason_groups.items()):
                print(f"    Reason: {reason} ({len(reason_failures)} runs)")
                for failure in reason_failures:
                    print(f"      {failure['id']} | {failure['state']} | {failure['url']}")
    
    print(f"\nAll {total_failures} failed run URLs have been saved to {output_file}")
    
    # Delete runs if requested
    if delete:
        if dry_run:
            print("\n=== DRY RUN: The following runs would be deleted ===")
        else:
            print("\n=== DELETING RUNS (THIS CANNOT BE UNDONE) ===")
            
            # Login to wandb if needed
            try:
                wandb.login()
            except Exception as e:
                print(f"Error logging in to wandb: {e}")
                return failed_runs
        
        # Count by task for progress reporting
        delete_count = 0
        api = wandb.Api()
        
        for task, failures in task_failures.items():
            print(f"\nTask: {task} - {len(failures)} runs to delete")
            
            for failure in failures:
                run_id = failure["id"]
                url = failure["url"]
                
                if dry_run:
                    print(f"Would delete: {run_id} - {url}")
                else:
                    try:
                        print(f"Deleting run {run_id}... ", end="", flush=True)
                        run = api.run(f"{entity}/{project}/{run_id}")
                        run.delete()
                        print("SUCCESS")
                        delete_count += 1
                        
                        # Add a small delay to avoid rate limiting
                        time.sleep(delay)
                    except Exception as e:
                        print(f"FAILED: {e}")
        
        if not dry_run:
            print(f"\nDeleted {delete_count} of {total_failures} failed runs")
        else:
            print(f"\nDRY RUN: Would have deleted {total_failures} failed runs")
            print("To actually delete, run with --delete --no-dry-run")
    
    return failed_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and optionally delete failed runs from encoder_sweep_results.json")
    parser.add_argument("--results-file", default="encoder_sweep_results.json", help="Path to results file")
    parser.add_argument("--output-file", default="failed_run_urls.txt", help="Path to output file for URLs")
    parser.add_argument("--entity", default="mmarone-jhu", help="WandB entity")
    parser.add_argument("--project", default="encoder_glue_sweeps_fixed_save", help="WandB project")
    parser.add_argument("--tasks", nargs="+", help="Filter by specific tasks (e.g., mnli cola)")
    parser.add_argument("--sizes", nargs="+", help="Filter by model sizes (e.g., tiny base huge)")
    parser.add_argument("--delete", action="store_true", help="Delete the failed runs")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually delete runs (default is dry run)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between deletion requests in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Determine if we're in dry-run mode or not
    dry_run = not args.no_dry_run
    
    extract_and_delete_failed_runs(
        results_file=args.results_file,
        output_file=args.output_file,
        entity=args.entity,
        project=args.project,
        tasks=args.tasks,
        model_sizes=args.sizes,
        delete=args.delete,
        dry_run=dry_run,
        delay=args.delay
    )
    # python gather_failed_run_urls.py --delete