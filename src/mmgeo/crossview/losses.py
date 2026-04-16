"""Loss functions for cross-view contrastive learning.

Implements symmetric InfoNCE (Sample4Geo, Section 3.1):
- Each positive pair is contrasted against all other samples in the batch
- Loss is computed in both directions (ground→sat and sat→ground) and averaged
- Temperature parameter can be learnable (as in CLIP/Sample4Geo)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE loss for cross-view contrastive learning.

    Given a batch of N ground embeddings and N satellite embeddings where
    pair (i, i) is a positive match, computes cross-entropy loss treating
    all other pairs as negatives — in both directions.

    Parameters
    ----------
    temperature : float
        Initial temperature for scaling logits. Lower = sharper distribution.
    learnable_temp : bool
        If True, temperature is a learnable parameter (as in Sample4Geo).
    label_smoothing : float
        Label smoothing for cross-entropy (Sample4Geo uses 0.1).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if learnable_temp:
            # Store log(temperature) for numerical stability
            self.log_temp = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_temp", torch.tensor(temperature).log())
        self.label_smoothing = label_smoothing

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(
        self,
        ground_embeds: torch.Tensor,
        sat_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Parameters
        ----------
        ground_embeds : torch.Tensor, shape (B, D)
            L2-normalized ground image embeddings.
        sat_embeds : torch.Tensor, shape (B, D)
            L2-normalized satellite image embeddings.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Similarity matrix: (B, B)
        # Since embeddings are L2-normalized, dot product = cosine similarity
        logits = ground_embeds @ sat_embeds.T / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(len(logits), device=logits.device)

        # Cross-entropy in both directions
        loss_g2s = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_s2g = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)

        return (loss_g2s + loss_s2g) / 2
