"""Dual branch fusion network and its single modality variants."""
from __future__ import annotations

import torch
from torch import nn


class TabularEncoder(nn.Module):
    """Dense network for the clinical record."""

    def __init__(self, in_features: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, width = [], in_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.net = nn.Sequential(*layers)
        self.out_features = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNImageEncoder(nn.Module):
    """Compact residual CNN that turns an embryo image into an embedding."""

    def __init__(self, embedding: int = 128, widths: tuple[int, ...] = (24, 48, 96, 160)):
        super().__init__()
        blocks, channels = [], 1
        for width in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            ))
            channels = width
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, embedding), nn.ReLU())
        self.out_features = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.blocks(x))


class ViTImageEncoder(nn.Module):
    """Vision Transformer image encoder from Hugging Face Transformers.

    With ``pretrained_name`` set, weights are downloaded from the Hub and the
    single channel input is repeated to three channels. Otherwise a small ViT
    is initialised from a configuration and trained from scratch.
    """

    def __init__(self, embedding: int, image_size: int, vit_cfg: dict):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        self.repeat_channels = bool(vit_cfg.get("pretrained_name"))
        if self.repeat_channels:
            self.vit = ViTModel.from_pretrained(vit_cfg["pretrained_name"], add_pooling_layer=False)
        else:
            config = ViTConfig(image_size=image_size, patch_size=vit_cfg["patch_size"], num_channels=1, hidden_size=vit_cfg["hidden_size"],
                               num_hidden_layers=vit_cfg["layers"], num_attention_heads=vit_cfg["heads"],
                               intermediate_size=2 * vit_cfg["hidden_size"], hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.0)
            self.vit = ViTModel(config, add_pooling_layer=False)
        self.project = nn.Sequential(nn.Linear(self.vit.config.hidden_size, embedding), nn.ReLU())
        self.out_features = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeat_channels:
            x = x.repeat(1, 3, 1, 1)
        return self.project(self.vit(pixel_values=x).last_hidden_state[:, 0])


class FusionNet(nn.Module):
    """Tabular branch and image branch, concatenated before the classification head.

    ``mode`` selects ``tabular``, ``image`` or ``fusion``. In fusion mode a
    branch can be hidden at random during training (modality dropout), which
    stops the head from leaning on one view and lets the model score a case
    when a modality is unavailable at inference.
    """

    def __init__(self, mode: str, n_tabular: int, image_size: int, model_cfg: dict, image_encoder: str = "cnn"):
        super().__init__()
        if mode not in {"tabular", "image", "fusion"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode, self.modality_dropout = mode, float(model_cfg.get("modality_dropout", 0.0))
        self.tabular = TabularEncoder(n_tabular, list(model_cfg["tabular_hidden"]), model_cfg["dropout"]) if mode != "image" else None
        self.image = None
        if mode != "tabular":
            if image_encoder == "cnn":
                self.image = CNNImageEncoder(model_cfg["image_embedding"])
            elif image_encoder == "vit":
                self.image = ViTImageEncoder(model_cfg["image_embedding"], image_size, dict(model_cfg["vit"]))
            else:
                raise ValueError(f"Unknown image encoder: {image_encoder}")
        width = (self.tabular.out_features if self.tabular else 0) + (self.image.out_features if self.image else 0)
        self.head = nn.Sequential(nn.Linear(width, model_cfg["fusion_hidden"]), nn.ReLU(), nn.Dropout(model_cfg["dropout"]),
                                  nn.Linear(model_cfg["fusion_hidden"], 1))

    def forward(self, tabular: torch.Tensor, image: torch.Tensor, hide: str | None = None) -> torch.Tensor:
        """Return one logit per case. ``hide`` may be ``tabular`` or ``image`` to ablate a branch."""
        parts = []
        if self.tabular is not None:
            parts.append(self.tabular(tabular))
        if self.image is not None:
            parts.append(self.image(image))
        if self.mode == "fusion":
            if self.training and self.modality_dropout > 0:
                draw = torch.rand(len(tabular), 1, device=tabular.device)
                parts[0] = parts[0] * (draw > self.modality_dropout)
                parts[1] = parts[1] * (draw < 1 - self.modality_dropout)
            if hide == "tabular":
                parts[0] = torch.zeros_like(parts[0])
            elif hide == "image":
                parts[1] = torch.zeros_like(parts[1])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)
