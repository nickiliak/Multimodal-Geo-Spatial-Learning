#!/bin/bash
#BSUB -J crossview_eval_zeroshot_384
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 08:00
#BSUB -o logs/crossview_eval_zeroshot_384_%J.out
#BSUB -e logs/crossview_eval_zeroshot_384_%J.err

mkdir -p logs eval_results
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

echo "Protocol: Option A zero-shot (convnext_base.fb_in22k_ft_in1k_384, 384px, timm norm)"
echo "Protocol: --no-pool (18,689 individual queries, paper-comparable)"

uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_384_zeroshot.yaml \
    --pretrained-only \
    --no-pool \
    --landmark-agg max \
    --save-checkpoint "checkpoints/zeroshot_convnext_base_384.pt" \
    --output "eval_results/zeroshot_convnext_base_384_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
