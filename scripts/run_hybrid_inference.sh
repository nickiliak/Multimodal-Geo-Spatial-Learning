#!/bin/bash
#BSUB -J hybrid_infer
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=24GB]"
#BSUB -W 24:00
#BSUB -o logs/hybrid_infer_%J.out
#BSUB -e logs/hybrid_infer_%J.err

set -e

mkdir -p logs outputs/hybrid outputs/hybrid_cache
cd ~/Multimodal-Geo-Spatial-Learning || { echo "Project directory not found"; exit 1; }

echo "--------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------"
nvidia-smi

echo ">>> Syncing environment with uv..."
uv sync

echo ">>> Validating PyTorch CUDA..."
uv run --no-sync python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA unavailable'
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ">>> Running hybrid inference sweep..."
uv run --no-sync python scripts/run_hybrid_inference.py \
    --data-root data/MML_Data \
    --ckpt models/zeroshot_convnext_base_384.pt\
    --cache-dir outputs/hybrid_cache \
    --output outputs/hybrid/results_${LSB_JOBID}.json \
    --radii-km 10,inf \
    --index-modes  full \
    --query-modes  all \
    --fallbacks fallback_full \
    --num-workers 6 \
    --batch-size 128

echo "Job finished at $(date)"
