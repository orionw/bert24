#!/bin/bash

# Create and activate a new virtual environment
# python -m venv .venv
# source .venv/bin/activate

# # Install uv if not already installed
# pip install uv

# Main dependencies installation
uv pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.2 \
    transformers==4.44.1 \
    datasets==2.21.0 \
    evaluate==0.4.1 \
    huggingface_hub==0.24.6 \
    wandb==0.16.6 \
    torchmetrics==1.4.0.post0 \
    einops==0.8.0 \
    pytorch-ranger==0.1.1 \
    torch-optimizer==0.3.0 \
    safetensors==0.4.4 \
    tokenizers==0.19.1 \
    pillow==10.3.0 \
    tqdm==4.66.5 \
    psutil==6.0.0 \
    pyyaml==6.0.2 \
    omegaconf==2.3.0 \
    filelock==3.15.4 \
    boto3==1.35.2 \
    google-cloud-storage==2.10.0 \
    azure-storage-blob==12.22.0 \
    ninja==1.11.1.1 \
    mosaicml==0.24.1 \
    mosaicml-cli==0.6.41 \
    mosaicml-streaming==0.8.0 \
    pytest==8.3.2 \
    pytest-xdist==3.6.1

# Note: CUDA libraries should be installed separately at the system level
# The script assumes CUDA 12.4 is installed on the system