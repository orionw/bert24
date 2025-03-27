import wandb
import json
import os
from collections import defaultdict
import time
import argparse
from datetime import datetime, timedelta

# Task-specific metrics from your creation script
METRIC_MAP = {
    "mnli": "MulticlassAccuracy",
    "cola": "MulticlassMatthewsCorrCoef",
    "qnli": "MulticlassAccuracy",
    "qqp": "MulticlassAccuracy",
    "sst2": "MulticlassAccuracy",
    "mrpc": "MulticlassAccuracy",
    "stsb": "SpearmanCorrCoef",
}

class SweepResultTracker:
    def __init__(self, results_file="encoder_sweep_results.json", cache_file="sweep_cache.json", 
                 sweep_mapping_file="sweep_ids_all.json", entity="mmarone-jhu", project="encoder_glue_sweeps_fixed_save"):
        self.results_file = results_file
        self.cache_file = cache_file
        self.entity = entity
        self.project = project
        
        # Load sweep ID mapping
        self.sweep_mapping = self._load_sweep_mapping(sweep_mapping_file)
        
        # Initialize cache and results
        self.cache = self._load_cache()
        self.results = self._load_results()
        
        # Track API call stats for optimization
        self.api_calls = 0
        self.cache_hits = 0

    def _load_sweep_mapping(self, mapping_file):
        """Load sweep ID mapping from file"""
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        
        # If file doesn't exist, use hardcoded mapping as fallback
        return {
            "encoder-very_tiny-mnli-42": "53y1fct4",
            "encoder-tiny-mnli-42": "fsw83dqp",
            "encoder-mini-mnli-42": "q5x8ynqm",
            "encoder-base-mnli-42": "s0hzsgup",
            "encoder-huge-mnli-42": "huyp2b2r",
            "encoder-very_tiny-cola-42": "5txs4nc2",
            "encoder-tiny-cola-42": "zacxyabj",
            "encoder-mini-cola-42": "z1p9b751",
            "encoder-base-cola-42": "a71d79wa",
            "encoder-huge-cola-42": "3ywazqfh",
            "encoder-very_tiny-qnli-42": "tpdqnjnq",
            "encoder-tiny-qnli-42": "ibrpfszp",
            "encoder-mini-qnli-42": "zs577nni",
            "encoder-base-qnli-42": "anh8cyxd",
            "encoder-huge-qnli-42": "fw2m9866",
            "encoder-very_tiny-qqp-42": "owus14tm",
            "encoder-tiny-qqp-42": "p4ikfcc5",
            "encoder-mini-qqp-42": "nnpt499f",
            "encoder-base-qqp-42": "pbtkf0g0",
            "encoder-huge-qqp-42": "dnhfa4ga",
            "encoder-very_tiny-sst2-42": "uzp7u6h8",
            "encoder-tiny-sst2-42": "as9zmsth",
            "encoder-mini-sst2-42": "cfdpydrf",
            "encoder-base-sst2-42": "qrad1lh8",
            "encoder-huge-sst2-42": "ujc2lafv"
        }

    def _load_cache(self):
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {"last_updated": {}, "run_metrics": {}, "last_cache_write": time.time()}

    def _save_cache(self):
        """Save cache to file"""
        self.cache["last_cache_write"] = time.time()
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _load_results(self):
        """Load existing results from file if it exists"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_results(self):
        """Save results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def parse_config_name(self, config_name):
        """Parse config name to get model type, size and task"""
        parts = config_name.split('-')
        if len(parts) >= 3:
            model_type = parts[0]
            model_size = parts[1]
            task = parts[2]
            return model_type, model_size, task
        return None, None, None

    def get_metric_name(self, task):
        """Get the metric name for a task based on METRIC_MAP"""
        return f"metrics/glue_{task}/{METRIC_MAP.get(task, 'MulticlassAccuracy')}"

    def _should_refresh_sweep(self, sweep_id, config_name):
        """Determine if we should refresh sweep data based on cache"""
        if sweep_id not in self.cache["last_updated"]:
            return True
            
        # Check if it's been more than 1 hour since last check for running sweeps
        last_updated = self.cache["last_updated"].get(sweep_id, 0)
        time_since_update = time.time() - last_updated
        
        # If the config is in results, check less frequently (every 4 hours)
        if config_name in self.results:
            return time_since_update > 14400  # 4 hours
        
        # Otherwise check more frequently (every 30 minutes)
        return time_since_update > 1800  # 30 minutes

    def _detect_failures(self, run, task, metric_name):
        """Detect if a run has failed and determine the reason"""
        run_score = run["metrics"].get(metric_name, None)
        
        # Default values
        is_failed = False
        failure_reason = None
        
        # Check #1: Missing metric score
        if run_score is None:
            # For running jobs, consider it failed if running for a long time without metrics
            if run["state"] == "running":
                run_timestamp = run.get("created_at", None)
                if run_timestamp is not None and isinstance(run_timestamp, str):
                    try:
                        created_time = datetime.fromisoformat(run_timestamp.replace('Z', '+00:00'))
                        # If running for over 1 hour with no metrics, likely failed
                        if datetime.now() - created_time > timedelta(hours=1):
                            is_failed = True
                            failure_reason = "Running >1hr with no metrics (likely failed)"
                    except ValueError:
                        pass
            # For finished jobs, definitely consider it failed
            elif run["state"] == "finished":
                is_failed = True
                failure_reason = f"Missing metric: {metric_name}"
        
        # Check #2: Run state indicates failure
        if run["state"] in ["crashed", "failed", "killed"]:
            is_failed = True
            failure_reason = f"Run {run['state']}"
        
        # Check #3: COLA special case - negative scores are valid but indicate poor performance
        if task == "cola" and run_score is not None and run_score < 0:
            # For COLA we'll flag it in the report but not count as failed
            failure_reason = "Negative score (poor performance but not failure)"
            
        # Check #4: Detect runs that might be stuck (running for too long)
        if not is_failed and run["state"] == "running":
            run_timestamp = run.get("created_at", None)
            if run_timestamp is not None and isinstance(run_timestamp, str):
                try:
                    created_time = datetime.fromisoformat(run_timestamp.replace('Z', '+00:00'))
                    if datetime.now() - created_time > timedelta(hours=6):
                        is_failed = True
                        failure_reason = "Run stuck (running for >6 hours)"
                except ValueError:
                    pass
        
        # Check #5: Look for error indicators in metrics
        if not is_failed:
            for key in run["metrics"]:
                if "error" in key.lower() and run["metrics"][key] not in [0, False, None]:
                    is_failed = True
                    failure_reason = f"Error detected: {key}"
                    break
        
        # Check #6: Look for minimal metrics but incomplete run
        if not is_failed and run["state"] == "running" and len(run["metrics"]) <= 3:
            # If a running job has very few metrics, it might be stuck at initialization
            run_timestamp = run.get("created_at", None)
            if run_timestamp is not None and isinstance(run_timestamp, str):
                try:
                    created_time = datetime.fromisoformat(run_timestamp.replace('Z', '+00:00'))
                    if datetime.now() - created_time > timedelta(minutes=30):
                        is_failed = True
                        failure_reason = "Minimal metrics after 30min (likely stuck)"
                except ValueError:
                    pass
                    
        return is_failed, failure_reason, run_score

    def get_sweep_results(self, sweep_id, config_name, force_refresh=False):
        """Get all results from a sweep with caching"""
        model_type, model_size, task = self.parse_config_name(config_name)
        metric_name = self.get_metric_name(task)
        
        if not force_refresh and not self._should_refresh_sweep(sweep_id, config_name):
            self.cache_hits += 1
            cached_runs = self.cache.get("run_metrics", {}).get(sweep_id, [])
            if cached_runs:
                print(f"[CACHE HIT] Using cached data for {config_name}")
                
                # Process all runs
                all_runs = []
                best_score = None
                best_run_id = None
                
                for run in cached_runs:
                    # Detect failures
                    is_failed, failure_reason, run_score = self._detect_failures(run, task, metric_name)
                    
                    # Track best score
                    if run_score is not None and (best_score is None or run_score > best_score):
                        best_score = run_score
                        best_run_id = run["id"]
                    
                    # Create run data
                    run_data = {
                        "id": run["id"],
                        "name": run["name"],
                        "state": run["state"],
                        "score": run_score,
                        "is_failed": is_failed,
                        "failure_reason": failure_reason,
                        "all_metrics": run["metrics"],
                        "created_at": run.get("created_at", None),
                        "config": run.get("config", {})
                    }
                    all_runs.append(run_data)
                
                # Mark best run
                for run in all_runs:
                    run["is_best"] = run["id"] == best_run_id
                
                return {
                    "config": config_name,
                    "model_type": model_type,
                    "model_size": model_size,
                    "task": task,
                    "sweep_id": sweep_id,
                    "best_run": best_run_id,
                    "best_score": best_score,
                    "metric": metric_name,
                    "runs": all_runs,
                    "total_runs": len(all_runs),
                    "failed_runs": sum(1 for run in all_runs if run["is_failed"]),
                    "source": "cache"
                }
                
        # If we need to refresh or no valid cache, fetch from API
        print(f"[API CALL] Fetching data for {config_name} (sweep: {sweep_id})")
        self.api_calls += 1
        
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{self.entity}/{self.project}/{sweep_id}")
            
            # Save all run data to cache
            cached_runs = []
            
            for run in sweep.runs:
                try:
                    # Get the summary statistics for the run
                    summary = run.summary._json_dict
                    metrics = {}
                    
                    # Extract all metrics
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                    
                    # Save run data to cache
                    cached_runs.append({
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "metrics": metrics,
                        "created_at": run.created_at if hasattr(run, 'created_at') else None,
                        "config": run.config if hasattr(run, 'config') else {}
                    })
                except Exception as e:
                    print(f"Error processing run {run.id}: {e}")
            
            # Update cache
            if "run_metrics" not in self.cache:
                self.cache["run_metrics"] = {}
            self.cache["run_metrics"][sweep_id] = cached_runs
            self.cache["last_updated"][sweep_id] = time.time()
            
            # Optimize by only writing to disk periodically
            cache_age = time.time() - self.cache.get("last_cache_write", 0)
            if cache_age > 300:  # 5 minutes
                self._save_cache()
            
            # Process all runs
            all_runs = []
            best_score = None
            best_run_id = None
            
            for run in cached_runs:
                # Detect failures
                is_failed, failure_reason, run_score = self._detect_failures(run, task, metric_name)
                
                # Track best score
                if run_score is not None and (best_score is None or run_score > best_score):
                    best_score = run_score
                    best_run_id = run["id"]
                
                # Create run data
                run_data = {
                    "id": run["id"],
                    "name": run["name"],
                    "state": run["state"],
                    "score": run_score,
                    "is_failed": is_failed,
                    "failure_reason": failure_reason,
                    "all_metrics": run["metrics"],
                    "created_at": run.get("created_at", None),
                    "config": run.get("config", {})
                }
                all_runs.append(run_data)
            
            # Mark best run
            for run in all_runs:
                run["is_best"] = run["id"] == best_run_id
            
            return {
                "config": config_name,
                "model_type": model_type,
                "model_size": model_size,
                "task": task,
                "sweep_id": sweep_id,
                "best_run": best_run_id,
                "best_score": best_score,
                "metric": metric_name,
                "runs": all_runs,
                "total_runs": len(all_runs),
                "failed_runs": sum(1 for run in all_runs if run["is_failed"]),
                "source": "api",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching sweep {sweep_id}: {e}")
            return None

    def update_results(self, force_refresh=False, tasks=None, model_sizes=None):
        """Update results for all sweeps or filtered by tasks/sizes"""
        # Group configs by task for better sorting
        task_groups = defaultdict(list)
        for config_name, sweep_id in self.sweep_mapping.items():
            model_type, model_size, task = self.parse_config_name(config_name)
            
            # Apply filters if specified
            if tasks and task not in tasks:
                continue
            if model_sizes and model_size not in model_sizes:
                continue
                
            task_groups[task].append(config_name)
        
        # Get results for each sweep
        updated = False
        for task, configs in task_groups.items():
            print(f"\n=== Processing task: {task} ===")
            for config_name in sorted(configs):
                sweep_id = self.sweep_mapping[config_name]
                
                result = self.get_sweep_results(sweep_id, config_name, force_refresh)
                if result:
                    old_result = self.results.get(config_name, {})
                    old_score = old_result.get("best_score", None)
                    old_runs = len(old_result.get("runs", [])) if "runs" in old_result else 0
                    
                    self.results[config_name] = result
                    updated = True
                    
                    # Display update information
                    new_runs = len(result["runs"])
                    failed_runs = result["failed_runs"]
                    
                    # Handle None score values gracefully
                    score_display = f"{result['best_score']:.4f}" if result['best_score'] is not None else "N/A"
                    
                    if old_score is None:
                        print(f"Added new result for {config_name}: {score_display} ({new_runs} runs, {failed_runs} failed)")
                    else:
                        # Calculate score diff only if both scores are valid numbers
                        if result['best_score'] is not None and old_score is not None:
                            score_diff = f"({result['best_score'] - old_score:+.4f})" if result['best_score'] != old_score else ""
                        else:
                            score_diff = ""
                            
                        run_diff = f"(+{new_runs - old_runs})" if new_runs != old_runs else ""
                        print(f"Updated {config_name}: {score_display} {score_diff} | {new_runs} runs {run_diff}, {failed_runs} failed")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
        
        if updated:
            # Sort results by task and score
            for task in sorted(task_groups.keys()):
                print(f"\n=== Results for {task} ===")
                task_configs = task_groups[task]
                
                # Handle potential None scores when sorting
                def get_sort_key(x):
                    if x not in self.results:
                        return -float('inf')
                    score = self.results[x].get("best_score")
                    return score if score is not None else -float('inf')
                
                sorted_configs = sorted(
                    [c for c in task_configs if c in self.results],
                    key=get_sort_key,
                    reverse=True
                )
                
                # Print a summary of the sorted results
                for config in sorted_configs:
                    result = self.results[config]
                    model_size = result["model_size"]
                    total_runs = result["total_runs"]
                    failed_runs = result["failed_runs"]
                    
                    # Handle None score values gracefully
                    score_display = f"{result['best_score']:.4f}" if result['best_score'] is not None else "N/A"
                    
                    print(f"{model_size.ljust(10)}: {score_display} | {total_runs} runs, {failed_runs} failed")
                    
                    # Print individual run details if requested
                    if os.environ.get("SHOW_ALL_RUNS", "0") == "1":
                        for i, run in enumerate(sorted(result["runs"], key=lambda x: x["score"] if x["score"] is not None else -float('inf'), reverse=True)):
                            status = "FAILED" if run["is_failed"] else ("BEST" if run["is_best"] else "OK")
                            score = f"{run['score']:.4f}" if run["score"] is not None else "N/A"
                            reason = f" - {run['failure_reason']}" if run["is_failed"] and run.get("failure_reason") else ""
                            print(f"  {i+1:2d}. {run['id']} - {score} - {run['state']} - {status}{reason}")
            
            self._save_results()
            self._save_cache()  # Make sure cache is saved
            print(f"\nResults updated and saved to {self.results_file}")
            print(f"API calls: {self.api_calls}, Cache hits: {self.cache_hits}")
        else:
            print("\nNo new results to update")

    def print_run_details(self, config_name=None, sweep_id=None, show_failed_only=False):
        """Print detailed information about runs for a specific config or sweep"""
        if config_name and config_name in self.results:
            result = self.results[config_name]
            sweep_id = result["sweep_id"]
        elif sweep_id and any(r["sweep_id"] == sweep_id for r in self.results.values()):
            # Find the corresponding config
            for config, result in self.results.items():
                if result["sweep_id"] == sweep_id:
                    config_name = config
                    break
        else:
            print(f"No results found for the specified config or sweep ID")
            return
            
        result = self.results[config_name]
        
        print(f"\n=== Run Details for {config_name} (Sweep: {sweep_id}) ===")
        print(f"Task: {result['task']}, Model Size: {result['model_size']}")
        
        # Handle None score values gracefully
        score_display = f"{result['best_score']:.4f}" if result['best_score'] is not None else "N/A"
        print(f"Best Score: {score_display}, Total Runs: {result['total_runs']}, Failed: {result['failed_runs']}")
        
        # Sort runs by score (best first)
        sorted_runs = sorted(
            result["runs"], 
            key=lambda x: x["score"] if x["score"] is not None else -float('inf'), 
            reverse=True
        )
        
        # Filter if needed
        if show_failed_only:
            sorted_runs = [r for r in sorted_runs if r["is_failed"]]
            print(f"\nShowing {len(sorted_runs)} failed runs:")
        else:
            print(f"\nShowing all {len(sorted_runs)} runs:")
            
        # Print run details in table format
        print(f"\n{'ID'.ljust(10)} | {'Name'.ljust(20)} | {'State'.ljust(10)} | {'Score'.ljust(10)} | Status")
        print(f"{'-'*10} | {'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
        
        for run in sorted_runs:
            status = "FAILED" if run["is_failed"] else ("BEST" if run["is_best"] else "OK")
            score = f"{run['score']:.4f}" if run["score"] is not None else "N/A"
            reason = f" - {run['failure_reason']}" if run["is_failed"] and run.get("failure_reason") else ""
            print(f"{run['id'][:8].ljust(10)} | {run['name'][:20].ljust(20)} | {run['state'].ljust(10)} | {score.ljust(10)} | {status}{reason}")

    def print_status_counts(self):
        """Print status counts for each sweep"""
        print("\n=== Run Status Counts by Sweep ===")
        
        if not self.results:
            print("No results available. Run the script without --status-counts first to collect data.")
            return
            
        for config_name, result in self.results.items():
            # Check if the result has the expected keys
            if not all(key in result for key in ["sweep_id", "model_size", "task", "runs"]):
                print(f"Skipping {config_name}: missing required data")
                continue
                
            sweep_id = result["sweep_id"]
            model_size = result["model_size"]
            task = result["task"]
            runs = result.get("runs", [])
            
            if not runs:
                print(f"\n{config_name} ({sweep_id}):")
                print(f"  Model: {model_size}, Task: {task}")
                print(f"  No runs available")
                continue
            
            # Count run states
            state_counts = {}
            for run in runs:
                state = run.get("state", "unknown")
                if state not in state_counts:
                    state_counts[state] = 0
                state_counts[state] += 1
            
            # Count failure reasons
            failure_counts = {}
            for run in runs:
                if run.get("is_failed", False):
                    reason = run.get("failure_reason") or "Unknown"
                    if reason not in failure_counts:
                        failure_counts[reason] = 0
                    failure_counts[reason] += 1
            
            # Print counts
            print(f"\n{config_name} ({sweep_id}):")
            print(f"  Model: {model_size}, Task: {task}")
            print(f"  Total runs: {len(runs)}")
            
            print("  States:")
            for state, count in sorted(state_counts.items()):
                print(f"    {state}: {count}")
            
            if failure_counts:
                print("  Failure reasons:")
                for reason, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {reason}: {count}")
            else:
                print("  No failed runs detected")


def main():
    parser = argparse.ArgumentParser(description="Update and monitor wandb sweep results with individual run tracking")
    parser.add_argument("--force", action="store_true", help="Force refresh all sweeps")
    parser.add_argument("--tasks", nargs="+", help="Filter by specific tasks (e.g., mnli cola)")
    parser.add_argument("--sizes", nargs="+", help="Filter by model sizes (e.g., tiny base huge)")
    parser.add_argument("--results-file", default="encoder_sweep_results.json", help="Path to results file")
    parser.add_argument("--cache-file", default="sweep_cache.json", help="Path to cache file")
    parser.add_argument("--mapping-file", default="sweep_ids_all.json", help="Path to sweep mapping file")
    parser.add_argument("--entity", default="mmarone-jhu", help="WandB entity")
    parser.add_argument("--project", default="encoder_glue_sweeps_fixed_save", help="WandB project")
    parser.add_argument("--details", help="Show detailed run information for a specific config name")
    parser.add_argument("--sweep", help="Show detailed run information for a specific sweep ID")
    parser.add_argument("--failed-only", action="store_true", help="Show only failed runs when using --details or --sweep")
    parser.add_argument("--show-all-runs", action="store_true", help="Show all runs in the summary output")
    parser.add_argument("--status-counts", action="store_true", help="Show status counts for each sweep")
    
    args = parser.parse_args()
    
    # Set environment variable for detailed output
    if args.show_all_runs:
        os.environ["SHOW_ALL_RUNS"] = "1"
    
    # Initialize wandb
    try:
        wandb.login()
        tracker = SweepResultTracker(
            results_file=args.results_file,
            cache_file=args.cache_file,
            sweep_mapping_file=args.mapping_file,
            entity=args.entity,
            project=args.project
        )
        
        # Check if we're showing status counts
        if args.status_counts:
            tracker.print_status_counts()
        # Check if we're showing details for a specific config or sweep
        elif args.details or args.sweep:
            tracker.print_run_details(
                config_name=args.details, 
                sweep_id=args.sweep, 
                show_failed_only=args.failed_only
            )
        else:
            # Update results normally
            tracker.update_results(force_refresh=args.force, tasks=args.tasks, model_sizes=args.sizes)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()