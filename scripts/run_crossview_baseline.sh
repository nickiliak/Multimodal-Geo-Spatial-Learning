#!/bin/bash
#BSUB -J crossview_train
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/crossview_train_%J.out
#BSUB -e logs/crossview_train_%J.err

# Cross-view retrieval training on DTU HPC
# Submit with: bsub < scripts/run_crossview_train.sh

mkdir -p logs

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Ensure data symlink exists
bash scripts/setup_data.sh <<< "N"

# Install timm if not present
uv pip install timm --quiet

# Run training
uv run python -m mmgeo.crossview.train --config configs/crossview_baseline.yaml

echo "Job finished at $(date)"
