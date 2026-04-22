"""Cross-view retrieval model: shared-weight ConvNeXt encoder.

Following Sample4Geo, we use a single ConvNeXt backbone with shared weights
for both ground and satellite views. The backbone outputs L2-normalized
embeddings that are compared via cosine similarity.

Key design choice: shared weights outperform separate encoders (Sample4Geo
Table 4), and the model is simpler with half the parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for the cross-view model. "
        "Install it with: uv pip install timm"
    )


class CrossViewModel(nn.Module):
    """Shared-weight image encoder for cross-view retrieval.

    Parameters
    ----------
    backbone : str
        timm model name. Recommended:
        - ``"convnext_tiny.fb_in22k"`` (28M params, fast iteration)
        - ``"convnext_base.fb_in22k"`` (88M params, Sample4Geo default)
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    embed_dim : int
        Output embedding dimension. If 0, uses backbone's native dim.
    """

    def __init__(
        self,
        backbone: str = "convnext_base.fb_in22k",
        pretrained: bool = True,
        embed_dim: int = 0,
    ) -> None:
        super().__init__()
        # num_classes=0 removes classification head, returns pooled features
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Optional projection head
        if embed_dim > 0 and embed_dim != self.feature_dim:
            self.proj = nn.Linear(self.feature_dim, embed_dim)
            self.embed_dim = embed_dim
        else:
            self.proj = nn.Identity()
            self.embed_dim = self.feature_dim

        print(
            f"[CrossViewModel] backbone={backbone}, "
            f"feature_dim={self.feature_dim}, embed_dim={self.embed_dim}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalized embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, H, W)

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            L2-normalized embeddings.
        """
        features = self.backbone(x)         # (B, feature_dim)
        embeddings = self.proj(features)     # (B, embed_dim)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference (no grad, eval mode)."""
        self.eval()
        return self.forward(images)
