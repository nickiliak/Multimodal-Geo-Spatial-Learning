"""Loss functions for cross-view contrastive learning.

Implements symmetric InfoNCE (Sample4Geo, Section 3.1):
- Each positive pair is contrasted against all other samples in the batch
- Loss is computed in both directions (ground→sat and sat→ground) and averaged
- Temperature parameter can be learnable (as in CLIP/Sample4Geo)

Also implements MultiPositiveInfoNCE (v3):
- Extends SymmetricInfoNCE to K ground images per satellite landmark
- Each satellite query treats all K ground views of its landmark as positives
- When K=1 reduces exactly to SymmetricInfoNCE
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


class MultiPositiveInfoNCE(nn.Module):
    """Symmetric InfoNCE with K ground images per satellite landmark (v3).

    Extends SymmetricInfoNCE to exploit multiple ground-level photos of the
    same landmark as multiple positives for the corresponding satellite query.
    When ``n_ground=1`` (K=1) the loss reduces exactly to SymmetricInfoNCE.

    Batch layout
    ------------
    ground_embeds : (B*K, D)  — K ground embeddings per landmark, ordered as
        [lm0_view0, lm0_view1, ..., lm0_view{K-1}, lm1_view0, ...]
    sat_embeds    : (B, D)    — one satellite embedding per landmark

    Loss directions
    ---------------
    s2g : satellite i → ground gallery (B*K items).
        Positives are columns i*K … i*K+K-1 (uniform soft label 1/K each).
    g2s : ground j → satellite gallery (B items).
        Each ground j has one satellite positive: column j // K.

    Parameters
    ----------
    temperature : float
    learnable_temp : bool
    label_smoothing : float
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if learnable_temp:
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
        BK = ground_embeds.shape[0]
        B = sat_embeds.shape[0]
        K = BK // B

        # Similarity matrix: sat_i vs all ground images
        # Shape: (B, B*K)
        sims = sat_embeds @ ground_embeds.T / self.temperature

        # ------------------------------------------------------------------
        # s2g direction: satellite i has K ground positives
        # Soft target: uniform 1/K over the K positives, 0 elsewhere
        # ------------------------------------------------------------------
        targets = torch.zeros(B, BK, device=sims.device)
        for i in range(B):
            targets[i, i * K : (i + 1) * K] = 1.0 / K

        # Apply label smoothing to the soft target distribution
        if self.label_smoothing > 0:
            targets = (1.0 - self.label_smoothing) * targets + self.label_smoothing / BK

        log_probs = F.log_softmax(sims, dim=1)
        loss_s2g = -(targets * log_probs).sum(dim=1).mean()

        # ------------------------------------------------------------------
        # g2s direction: ground j has one satellite positive (j // K)
        # Standard cross-entropy with hard labels
        # ------------------------------------------------------------------
        sat_labels = torch.arange(B, device=sims.device).repeat_interleave(K)
        loss_g2s = F.cross_entropy(
            sims.T, sat_labels, label_smoothing=self.label_smoothing
        )

        return (loss_g2s + loss_s2g) / 2
