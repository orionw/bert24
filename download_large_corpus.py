import os
import json
from huggingface_hub import hf_hub_download, list_repo_files
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob

def get_repo_files(repo_id, force_refresh=False):
    """
    Get list of files from repository, using cached version if available.
    
    Args:
        repo_id (str): Hugging Face repository ID
        force_refresh (bool): If True, ignore cache and fetch fresh list
    
    Returns:
        list: List of files in the repository
    """
    # Create cache directory if it doesn't exist
    os.makedirs(".cache", exist_ok=True)
    
    # Cache file path
    cache_path = os.path.join(".cache", f"{repo_id.replace('/', '_')}_files.json")
    
    # If not forcing refresh and cache exists, use it
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_files = json.load(f)
            print(f"Using cached repository file list from {cache_path}")
            return cached_files
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print(f"Error reading cache: {e}")
            # If there's an error reading cache, continue to fetch from API
    
    # Fetch from HF API
    try:
        print("Fetching repository file list from Hugging Face...")
        repo_files = list_repo_files(repo_id, repo_type="dataset")
        print(f"Found {len(repo_files)} files in repository")
        
        # Save to cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(repo_files, f)
            print(f"Repository file list cached to {cache_path}")
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print(f"Error writing cache: {e}")
        
        return repo_files
        
    except Exception as e:
        print(f"Error accessing repository: {e}")
        raise

def download_hf_files(output_dir="text-version-data-1k-chunks", 
                     repo_id="allenai/dolmino-mix-1124",
                     force_refresh=False,
                     workers=5):
    """
    Download all files from a Hugging Face repository, skipping existing files.
    
    Args:
        output_dir (str): Directory to save downloaded files
        repo_id (str): Hugging Face repository ID
        force_refresh (bool): If True, fetch fresh repository file list instead of using cache
        workers (int): Number of parallel download workers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get repository files (from cache or API)
    try:
        repo_files = get_repo_files(repo_id, force_refresh)
    except Exception as e:
        print(f"Failed to get repository file list: {e}")
        import traceback
        print(traceback.print_exc())
        return
    
    # Get list of already downloaded files, recursively looking for *.jsonl* files
    existing_files = set(["/".join(item.split("/")[1:]) for item in glob.glob(f"{output_dir}/**/*.jsonl*", recursive=True)])
    files_to_download = [f for f in repo_files if (f not in existing_files and ".log" not in f)]
    
    if not files_to_download:
        print("All files have already been downloaded!")
        return
    
    print(f"Downloading {len(files_to_download)} new files...")
    
    def download_file(filename):
        if os.path.exists(os.path.join(output_dir, filename)):
            return True
        try:
            print("Downloading...", filename, output_dir)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print(f"Error downloading {filename}: {e}")
            return False
    
    # Download files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(download_file, files_to_download),
            total=len(files_to_download),
            desc="Downloading files"
        ))
    
    # Print summary
    successful = sum(results)
    failed = len(files_to_download) - successful
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful} files")
    if failed > 0:
        print(f"Failed to download: {failed} files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download files from a Hugging Face repository")
    parser.add_argument("--output-dir", default="text-version-data-1k-chunks",
                      help="Directory to save downloaded files")
    parser.add_argument("--force-refresh", action="store_true",
                      help="Force refresh of repository file list instead of using cache")
    parser.add_argument("--workers", type=int, default=20,
                      help="Number of parallel download workers (default: 5)")
    parser.add_argument("--repo-id", type=str, default="orionweller/tokenized_validation_1k",
                        help="Hugging Face repository ID")
    
    args = parser.parse_args()
    download_hf_files(
        args.output_dir,
        repo_id=args.repo_id,
        force_refresh=args.force_refresh,
        workers=args.workers
    )