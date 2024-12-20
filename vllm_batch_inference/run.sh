#!/bin/bash

export HOME=$_CONDOR_SCRATCH_DIR
export HF_HOME=$_CONDOR_SCRATCH_DIR/huggingface

# If your job requests a single GPU, setting `CUDA_VISIBLE_DEVICES=0` ensures the system uses the correct device name and avoids errors. For multiple GPUs, set it accordingly (e.g., `0,1` for two GPUs).
export CUDA_VISIBLE_DEVICES=0

echo "Running job on `hostname`"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

echo "Setting up environment variables"
source .env

python3 batch_inference.py
echo "Job completed"
