#!/bin/sh
### LSF Queue Options
#BSUB -q gpuv100
#BSUB -J geoclip_finetuned
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

set -e

echo "--------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------"

export PATH="$HOME/.local/bin:$PATH"
cd ~/Multimodal-Geo-Spatial-Learning || { echo "Project directory not found"; exit 1; }

nvidia-smi

echo ">>> Syncing environment with uv..."
uv sync

echo ">>> Validating PyTorch CUDA..."
uv run --no-sync python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
assert torch.cuda.is_available(), 'CUDA unavailable'
" || {
    echo "CRITICAL: CUDA unavailable in Python environment."
    exit 1
}

if [ ! -f models/best_geoclip_baseline.pth ]; then
    echo "CRITICAL: checkpoint models/best_geoclip_baseline.pth not found. Run training first."
    exit 1
fi

echo ">>> Running GeoClip fine-tuned evaluation notebook..."
uv run --no-sync jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/team/04_geoclip_finetuned.ipynb

echo "Done."
