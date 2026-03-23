from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


@dataclass(frozen=True)
class EdgeAdapterArchitecture:
    obs_encoding_size: int
    seq_len: int
    mha_num_attention_layers: int
    mha_ff_dim_factor: int
    action_chunk_size: int
    action_dim: int


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_torch_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return strip_module_prefix(payload)


def infer_edge_adapter_architecture(
    state_dict: dict[str, torch.Tensor],
    action_chunk_fallback: int = 8,
) -> EdgeAdapterArchitecture:
    if "decoder.positional_encoding.pos_enc" not in state_dict:
        raise KeyError("Missing key: decoder.positional_encoding.pos_enc")
    pos_enc = state_dict["decoder.positional_encoding.pos_enc"]
    seq_len = int(pos_enc.shape[1])
    obs_encoding_size = int(pos_enc.shape[2])

    layer_indices: list[int] = []
    for key in state_dict:
        match = re.match(r"decoder\.sa_decoder\.layers\.(\d+)\.", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise KeyError("No decoder.sa_decoder.layers.* keys found")
    mha_num_attention_layers = max(layer_indices) + 1

    ff_dim = int(state_dict["decoder.sa_layer.linear1.weight"].shape[0])
    if ff_dim % obs_encoding_size != 0:
        raise ValueError(f"linear1 dim {ff_dim} is not divisible by embed dim {obs_encoding_size}")
    mha_ff_dim_factor = ff_dim // obs_encoding_size

    action_out = int(state_dict["action_predictor.6.weight"].shape[0])
    action_chunk_size = seq_len - 2 if seq_len > 2 else action_chunk_fallback
    if action_out % action_chunk_size != 0:
        action_chunk_size = action_chunk_fallback
    if action_out % action_chunk_size != 0:
        raise ValueError(f"Cannot infer action chunk from output dim {action_out}")
    action_dim = action_out // action_chunk_size

    return EdgeAdapterArchitecture(
        obs_encoding_size=obs_encoding_size,
        seq_len=seq_len,
        mha_num_attention_layers=mha_num_attention_layers,
        mha_ff_dim_factor=mha_ff_dim_factor,
        action_chunk_size=action_chunk_size,
        action_dim=action_dim,
    )


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int) -> None:
        super().__init__()
        pos_enc = torch.zeros(seq_len, embed_dim)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_enc[:, : x.shape[1], :]


class MultiLayerDecoderTrans(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        nhead: int,
        num_layers: int,
        ff_dim_factor: int,
    ) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, seq_len=seq_len)
        ff_dim = embed_dim * ff_dim_factor
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(tokens)
        return self.sa_decoder(x)


class EdgeAdapter(nn.Module):
    def __init__(
        self,
        obs_encoding_size: int,
        mha_num_attention_heads: int,
        mha_num_attention_layers: int,
        mha_ff_dim_factor: int,
        action_chunk_size: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = int(obs_encoding_size)
        self.action_chunk_size = int(action_chunk_size)
        self.action_dim = int(action_dim)

        self.cat_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        self.num_cat_features = int(self.cat_encoder._fc.in_features)
        self.num_obs_features = int(self.obs_encoder._fc.in_features)

        self.compress_obs_enc: nn.Module
        self.compress_cat_enc: nn.Module
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        if self.num_cat_features != self.obs_encoding_size:
            self.compress_cat_enc = nn.Linear(self.num_cat_features, self.obs_encoding_size)
        else:
            self.compress_cat_enc = nn.Identity()

        self.decoder = MultiLayerDecoderTrans(
            embed_dim=self.obs_encoding_size,
            seq_len=self.action_chunk_size + 2,
            nhead=int(mha_num_attention_heads),
            num_layers=int(mha_num_attention_layers),
            ff_dim_factor=int(mha_ff_dim_factor),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_chunk_size * self.action_dim),
        )

    @staticmethod
    def _encode_image(encoder: EfficientNet, image: torch.Tensor) -> torch.Tensor:
        encoding = encoder.extract_features(image)
        encoding = encoder._avg_pooling(encoding)
        if encoder._global_params.include_top:
            encoding = encoding.flatten(start_dim=1)
            encoding = encoder._dropout(encoding)
        return encoding

    def forward(
        self,
        obs_img: torch.Tensor,
        past_img: torch.Tensor,
        vla_feature: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.encode_fused(obs_img, past_img, vla_feature)
        return self.predict_action_from_fused(fused)

    def encode_fused(
        self,
        obs_img: torch.Tensor,
        past_img: torch.Tensor,
        vla_feature: torch.Tensor,
    ) -> torch.Tensor:
        cat_img = torch.cat((obs_img, past_img), dim=1)

        cat_encoding = self._encode_image(self.cat_encoder, cat_img)
        cat_encoding = self.compress_cat_enc(cat_encoding)

        obs_encoding = self._encode_image(self.obs_encoder, obs_img)
        obs_encoding = self.compress_obs_enc(obs_encoding)

        tokens = torch.cat((vla_feature, obs_encoding.unsqueeze(1), cat_encoding.unsqueeze(1)), dim=1)
        tokens = self.decoder(tokens)[:, -2:-1, :]
        return tokens.reshape(tokens.shape[0], -1)

    def predict_action_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 3 and fused.shape[1] == 1:
            fused = fused.reshape(fused.shape[0], -1)
        if fused.ndim != 2:
            raise ValueError(f"Expected fused latent shape [B, D] or [B, 1, D], got {tuple(fused.shape)}")
        action_pred = self.action_predictor(fused)
        return action_pred.reshape(fused.shape[0], self.action_chunk_size, self.action_dim)


class EdgeAdapterFusedBackbone(nn.Module):
    def __init__(self, edge_adapter: EdgeAdapter) -> None:
        super().__init__()
        self.edge_adapter = edge_adapter

    def forward(
        self,
        obs_img: torch.Tensor,
        past_img: torch.Tensor,
        vla_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_adapter.encode_fused(obs_img, past_img, vla_feature).unsqueeze(1)


class EdgeAdapterActionHead(nn.Module):
    def __init__(self, edge_adapter: EdgeAdapter) -> None:
        super().__init__()
        self.edge_adapter = edge_adapter

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.edge_adapter.predict_action_from_fused(fused)


def build_edge_adapter_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    mha_num_attention_heads: int = 4,
    action_chunk_fallback: int = 8,
    strict: bool = True,
) -> tuple[EdgeAdapter, EdgeAdapterArchitecture, list[str], list[str]]:
    arch = infer_edge_adapter_architecture(state_dict, action_chunk_fallback=action_chunk_fallback)
    model = EdgeAdapter(
        obs_encoding_size=arch.obs_encoding_size,
        mha_num_attention_heads=mha_num_attention_heads,
        mha_num_attention_layers=arch.mha_num_attention_layers,
        mha_ff_dim_factor=arch.mha_ff_dim_factor,
        action_chunk_size=arch.action_chunk_size,
        action_dim=arch.action_dim,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Edge adapter checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, arch, list(missing), list(unexpected)


def load_edge_adapter_from_hf_snapshot(
    hf_dir: str | Path,
    checkpoint_name: str = "shead--750000_checkpoint.pt",
    mha_num_attention_heads: int = 4,
    action_chunk_fallback: int = 8,
    strict: bool = True,
) -> tuple[EdgeAdapter, EdgeAdapterArchitecture, list[str], list[str]]:
    hf_path = Path(hf_dir).expanduser().resolve()
    ckpt_path = hf_path / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = load_torch_state_dict(ckpt_path)
    return build_edge_adapter_from_state_dict(
        state_dict=state_dict,
        mha_num_attention_heads=mha_num_attention_heads,
        action_chunk_fallback=action_chunk_fallback,
        strict=strict,
    )
