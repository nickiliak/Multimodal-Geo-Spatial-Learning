#!/bin/bash
#BSUB -J crossview_train
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# Uncomment to require a 32GB V100 (enables batch_size=128+):
# #BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=24GB]"
#BSUB -W 24:00
#BSUB -o logs/crossview_train_%J.out
#BSUB -e logs/crossview_train_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

uv run python -m mmgeo.crossview.train --config configs/crossview_baseline.yaml

echo "Job finished at $(date)"