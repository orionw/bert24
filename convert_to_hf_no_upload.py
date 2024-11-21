#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import requests
from huggingface_hub import HfApi, create_repo, hf_hub_download
from composer.models import write_huggingface_pretrained_from_composer_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Composer checkpoint to HuggingFace format'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the Composer checkpoint file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./hf-save-pretrained-output',
        help='Temporary directory for HuggingFace format conversion'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert checkpoint to HuggingFace format
    print(f"Converting checkpoint {args.checkpoint_path} to HuggingFace format...")
    write_huggingface_pretrained_from_composer_checkpoint(
        args.checkpoint_path,
        args.output_dir
    )
    
    print(f"Conversion complete. Files saved to {args.output_dir}")

if __name__ == '__main__':
    main()

    """
     ~/miniconda3/envs/mosiacbert/bin/python /home/oweller2/my_scratch/retrieval_pretraining/bert24/retrieval_pretraining/scripts/convert_to_hf_no_upload.py --checkpoint_path "/home/oweller2/my_scratch/retrieval_pretraining/bert24/models_pythia_like/decoder_no_packing/ep0-ba5996-rank0.pt"
    """