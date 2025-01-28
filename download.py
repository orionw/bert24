import argparse
from huggingface_hub import snapshot_download
import glob
import os

def download_files(repo_id, pattern, local_dir, workers=8, token=None):
    """
    Download files from a Hugging Face repository that match a specific pattern.

    Args:
        repo_id (str): The repository ID (e.g., 'username/repo-name')
        pattern (str): File pattern to match (e.g., '*.txt', 'model/*.safetensors')
        local_dir (str): Local directory to save the files
        token (str, optional): Hugging Face authentication token for private repos
    """
    try:
        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Use the allow_patterns parameter to filter files
        print(f"Downloading files from {repo_id} with pattern `{pattern}` to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=pattern,
            local_dir=local_dir,
            repo_type="dataset",
            max_workers=workers
        )

        print(f"Successfully downloaded files matching '{pattern}' to {local_dir}")

    except Exception as e:
        print(f"Error downloading files: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Download files from Hugging Face Hub')
    parser.add_argument('--repo', required=True, help='Repository ID (e.g., username/repo-name)')
    parser.add_argument('--pattern', required=True, help='File pattern to match (e.g., *.txt)')
    parser.add_argument('--output', required=True, help='Local directory to save files')
    parser.add_argument('--workers', default=8, type=int, help="Download workers")
    parser.add_argument('--token', help='Hugging Face authentication token', default=None)
    parser.add_argument('--dist', action='store_true', help="Use composer dist to block on other ranks")

    args = parser.parse_args()

    if args.dist:
        import composer.utils.dist as dist
        import torch
        device = dist.get_device()
        dist.initialize_dist(device, 300)
        with dist.run_local_rank_zero_first():
            download_files(args.repo, args.pattern, args.output, args.workers, args.token)
        # makes exit cleaner?
        dist.barrier()
        torch.distributed.destroy_process_group()
    else:
        download_files(args.repo, args.pattern, args.output, args.workers, args.token)

if __name__ == "__main__":
    main()
