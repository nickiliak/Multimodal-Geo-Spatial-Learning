#!/bin/bash
#BSUB -J crossview_eval_v3
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 08:00
#BSUB -o logs/crossview_eval_v3_%J.out
#BSUB -e logs/crossview_eval_v3_%J.err

mkdir -p logs eval_results
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

# v3 best checkpoint (ConvNeXt-Base fb_in22k_ft_in1k_384, 384px, epoch 36)
CKPT="checkpoints/crossview/cv_v3_base_20260429_055409/best.pt"

echo "Evaluating v3 checkpoint: $CKPT"
echo "Protocol: --no-pool (18,689 per-image queries) + --landmark-agg max (1,000 per-landmark)"

uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_v3.yaml \
    --checkpoint "$CKPT" \
    --no-pool \
    --landmark-agg max \
    --output "eval_results/eval_v3_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
