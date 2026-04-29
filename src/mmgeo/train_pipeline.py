"""
Training pipeline for finetuning hybrid pipeine, with frozen geoclip and trainable Sample4geo.
"""
#imports
from __future__ import annotations
#load geo from hugguingface
from mmgeo.geolocalizations.geoclip.geoclip_baseline import GeoClipBaseline
from mmgeo.crossview.model import CrossViewModel as Sample4Geo
#rest of the imports
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from mmgeo.crossview.dataset import (
    MMLCrossViewDataset,
    MMLImageDataset,
    UniqueLandmarkSampler,
    get_eval_transforms,
    get_train_transforms,
)
from mmgeo.crossview.evaluate import evaluate_crossview
from mmgeo.crossview.logging_utils import RunLogger
from mmgeo.crossview.losses import MultiPositiveInfoNCE, SymmetricInfoNCE
from mmgeo.crossview.model import CrossViewModel
from mmgeo.crossview.sampling import (
    HardNegativeBatchSampler,
    build_gps_neighbors,
    build_similarity_neighbors,
    compute_landmark_embeddings,
)
