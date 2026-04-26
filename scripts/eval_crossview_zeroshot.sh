#!/bin/bash
#BSUB -J crossview_eval_zeroshot
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 02:00
#BSUB -o logs/crossview_eval_zeroshot_%J.out
#BSUB -e logs/crossview_eval_zeroshot_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

echo "Protocol: zero-shot (ImageNet-22k pretrained weights only, no MMLandmarks training)"
echo "Protocol: --no-pool (18,689 individual queries, paper-comparable)"

uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base.yaml \
    --pretrained-only \
    --no-pool \
    --output "logs/eval_zeroshot_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
