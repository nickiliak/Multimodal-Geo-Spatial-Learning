#!/bin/bash
#BSUB -J crossview_eval_v4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -n 6
#BSUB -R "span[hosts=1] rusage[mem=32GB]"
#BSUB -W 02:00
#BSUB -o logs/crossview_eval_v4_%J.out
#BSUB -e logs/crossview_eval_v4_%J.err

mkdir -p logs eval_results
cd ~/Multimodal-Geo-Spatial-Learning

echo "Job started at $(date)"
echo "Running on $(hostname)"
nvidia-smi

# Set this to the v4 checkpoint directory after training completes
# e.g. checkpoints/crossview/cv_v4_base_20260514_XXXXXX/best.pt
CKPT=$(ls checkpoints/crossview/cv_v4_base_*/best.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No v4 checkpoint found. Check checkpoints/crossview/cv_v4_base_*/"
    exit 1
fi

echo "Evaluating v4 checkpoint: $CKPT"
echo "Protocol: --no-pool + --landmark-agg max (also computes mean in same pass)"

uv run python -m mmgeo.crossview.eval \
    --config configs/crossview_convnext_base_v4.yaml \
    --checkpoint "$CKPT" \
    --no-pool \
    --landmark-agg max \
    --output "eval_results/eval_v4_$(date +%Y%m%d_%H%M%S).json"

echo "Job finished at $(date)"
