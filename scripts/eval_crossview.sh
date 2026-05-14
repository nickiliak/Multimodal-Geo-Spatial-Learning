#!/bin/bash
#BSUB -J crossview_eval
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o logs/crossview_eval_%J.out
#BSUB -e logs/crossview_eval_%J.err

mkdir -p logs
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

# Path to the checkpoint to evaluate
CKPT="checkpoints/crossview/cv_v2_base_20260422_230539/best.pt"

echo "Evaluating checkpoint: $CKPT"
echo "Protocol: --no-pool (18,689 individual queries, paper-comparable)"

mkdir -p eval_results
uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base.yaml \
    --checkpoint "$CKPT" \
    --no-pool \
    --landmark-agg max \
    --output "eval_results/eval_v2_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
