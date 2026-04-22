#!/bin/bash
#BSUB -J crossview_base
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# Require a 32GB V100 — convnext_base @ 256px does NOT fit on 16GB at batch=64.
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/crossview_base_%J.out
#BSUB -e logs/crossview_base_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

uv run python -m mmgeo.crossview.train --config configs/crossview_convnext_base.yaml

echo "Job finished at $(date)"
