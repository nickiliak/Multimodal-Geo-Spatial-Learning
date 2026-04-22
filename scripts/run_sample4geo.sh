#!/bin/sh
### LSF Queue Options
#BSUB -q gpua100
#BSUB -J sample4geo
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -B
#BSUB -N
#BSUB -o outputs/jobs/Output_%J.out
#BSUB -e outputs/jobs/Output_%J.err

set -e

echo "--------------------------------------------------"
echo "Job ID: $LSB_JOBID | Node: $(hostname) | Date: $(date)"
echo "--------------------------------------------------"

# ---- Training hyperparameters (override defaults from model.py) ----
DATA_ROOT="data/MML_Data"
EPOCHS=40
BATCH_SIZE=32
LR=0.001
GPS_EPOCHS=10
DSS_REFRESH_EVERY=4

export PATH="$HOME/.local/bin:$PATH"
cd ~/Multimodal-Geo-Spatial-Learning || { echo "Project directory not found"; exit 1; }

# Load CUDA toolkit so PyTorch can find libcudnn etc.
module load cuda/12.1

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

echo ">>> Training Sample4Geo..."
echo "    epochs=$EPOCHS  batch_size=$BATCH_SIZE  lr=$LR"
echo "    gps_epochs=$GPS_EPOCHS  dss_refresh=$DSS_REFRESH_EVERY"

uv run --no-sync python -c "
from pathlib import Path
from mmgeo.geolocalizations.sample4geo.model import train

model = train(
    data_root=Path('${DATA_ROOT}'),
    epochs=${EPOCHS},
    batch_size=${BATCH_SIZE},
    lr=${LR},
    gps_epochs=${GPS_EPOCHS},
    dss_refresh_every=${DSS_REFRESH_EVERY},
    device='cuda',
)

import torch
torch.save(model.state_dict(), 'outputs/sample4geo.pt')
print('Model saved to outputs/sample4geo.pt')
"

echo "Done."
