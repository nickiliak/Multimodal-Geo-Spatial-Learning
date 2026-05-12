#!/bin/bash
#BSUB -J crossview_v4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# 224px + batch=64 + n_ground=2 fits comfortably in 32GB without AMP (~21GB peak)
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/crossview_v4_%J.out
#BSUB -e logs/crossview_v4_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

# Set to a checkpoint path to resume, or leave empty to start fresh.
RESUME=""

if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    echo "Resuming from checkpoint: $RESUME"
    uv run python -m mmgeo.crossview.train \
        --config configs/crossview_convnext_base_v4.yaml \
        --resume "$RESUME"
else
    uv run python -m mmgeo.crossview.train \
        --config configs/crossview_convnext_base_v4.yaml
fi

echo "Job finished at $(date)"
