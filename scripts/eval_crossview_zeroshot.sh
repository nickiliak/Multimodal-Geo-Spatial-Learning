#!/bin/bash
#BSUB -J crossview_eval_zeroshot
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 08:00
#BSUB -o logs/crossview_eval_zeroshot_%J.out
#BSUB -e logs/crossview_eval_zeroshot_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

echo "Protocol: zero-shot (ImageNet-22k + IN-1k 384px pretrained weights, no MMLandmarks training)"
echo "Protocol: --no-pool (18,689 individual queries, paper-comparable) + --landmark-agg max (per-landmark)"

mkdir -p eval_results
uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_384_zeroshot.yaml \
    --pretrained-only \
    --no-pool \
    --landmark-agg max \
    --output "eval_results/eval_zeroshot_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
