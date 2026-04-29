#!/bin/bash
#BSUB -J HybridFinetune
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
# Requires 32GB V100 — convnext_base @ 384px, GeoCLIP loaded alongside.
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/hybrid_finetune_%J.out
#BSUB -e logs/hybrid_finetune_%J.err

mkdir -p logs outputs/hybrid_cache
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

# Set RESUME to a checkpoint path to continue, or leave empty to start
# fresh from configs/hybrid.yaml model.pretrained_ckpt (models/finetuned.pt).
RESUME=""

if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    echo "Resuming from checkpoint: $RESUME"
    uv run python -m mmgeo.train_pipeline \
        --config configs/hybrid.yaml \
        --resume "$RESUME"
else
    uv run python -m mmgeo.train_pipeline \
        --config configs/hybrid.yaml
fi

echo "Job finished at $(date)"
