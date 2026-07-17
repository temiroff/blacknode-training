"""Compact vision-and-state action chunking transformer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - surfaced by package health and nodes
    torch = None
    nn = None


@dataclass(frozen=True)
class ActionChunkingConfig:
    state_dim: int
    action_dim: int
    camera_count: int
    chunk_size: int = 32
    hidden_dim: int = 256
    attention_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    dropout: float = 0.1
    image_tokens_per_camera: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionChunkingConfig":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


class ActionChunkingTransformer(nn.Module if nn is not None else object):
    """Predict a fixed horizon of normalized joint actions from images and state."""

    def __init__(self, config: ActionChunkingConfig) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required to create ActionChunkingTransformer")
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if config.camera_count < 1:
            raise ValueError("camera_count must be at least 1")
        if config.image_tokens_per_camera != 16:
            raise ValueError("this model currently requires image_tokens_per_camera=16")
        self.config = config
        hidden = config.hidden_dim
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.state_projection = nn.Linear(config.state_dim, hidden)
        self.camera_embedding = nn.Parameter(torch.zeros(config.camera_count, hidden))
        source_tokens = 1 + config.camera_count * config.image_tokens_per_camera
        self.source_position = nn.Parameter(torch.zeros(1, source_tokens, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.action_queries = nn.Parameter(torch.zeros(1, config.chunk_size, hidden))
        self.action_head = nn.Linear(hidden, config.action_dim)
        self.final_norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(self, qpos: Any, images: Any) -> Any:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch,camera,3,height,width]")
        batch, camera_count, channels, height, width = images.shape
        if camera_count != self.config.camera_count or channels != 3:
            raise ValueError(
                f"expected {self.config.camera_count} RGB cameras, got shape {tuple(images.shape)}"
            )
        encoded = self.vision(images.reshape(batch * camera_count, channels, height, width))
        encoded = encoded.flatten(2).transpose(1, 2)
        encoded = encoded.reshape(batch, camera_count, self.config.image_tokens_per_camera, self.config.hidden_dim)
        encoded = encoded + self.camera_embedding[None, :, None, :]
        image_tokens = encoded.reshape(batch, camera_count * self.config.image_tokens_per_camera, self.config.hidden_dim)
        state_token = self.state_projection(qpos).unsqueeze(1)
        source = torch.cat((state_token, image_tokens), dim=1) + self.source_position
        memory = self.encoder(source)
        queries = self.action_queries.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(self.final_norm(decoded))


def masked_l1_loss(prediction: Any, target: Any, is_pad: Any) -> Any:
    valid = (~is_pad).unsqueeze(-1).to(dtype=prediction.dtype)
    denominator = valid.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).abs() * valid).sum() / denominator
