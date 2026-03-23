"""Model weight I/O: load/save between PyTorch and Rust serde JSON format."""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .model import EntityAttention, PolicyNetV2Torch
from .normalize import RunningMeanStd, ValueNormalizer


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_linear(linear: nn.Linear, layer_dict: dict) -> None:
    in_dim = layer_dict["in_dim"]
    out_dim = layer_dict["out_dim"]
    w = torch.tensor(layer_dict["weights"], dtype=torch.float32).view(
        out_dim, in_dim
    )
    b = torch.tensor(layer_dict["biases"], dtype=torch.float32)
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)


def _load_layer_norm(ln: nn.LayerNorm, ln_dict: dict) -> None:
    ln.weight.data.copy_(
        torch.tensor(ln_dict["gamma"], dtype=torch.float32)
    )
    ln.bias.data.copy_(torch.tensor(ln_dict["beta"], dtype=torch.float32))


def _load_attention(attn: EntityAttention, attn_dict: dict) -> None:
    _load_linear(attn.w_q, attn_dict["w_q"])
    _load_linear(attn.w_k, attn_dict["w_k"])
    _load_linear(attn.w_v, attn_dict["w_v"])
    _load_linear(attn.w_o, attn_dict["w_o"])
    _load_layer_norm(attn.layer_norm, attn_dict["layer_norm"])


def load_model_weights(model: PolicyNetV2Torch, model_path: str) -> dict:
    """Load model weights from Rust JSON or flat binary format.

    Returns the raw JSON dict (or empty dict for binary format).
    """
    if model_path.endswith(".bin"):
        _load_model_weights_binary(model, model_path)
        return {}

    with open(model_path, "r") as f:
        d = json.load(f)

    _load_linear(model.ego_encoder, d["ego_encoder"])
    _load_linear(model.entity_encoder, d["entity_encoder"])
    _load_attention(model.attn, d["attn"])
    _load_attention(model.attn2, d["attn2"])
    _load_linear(model.fc1, d["fc1"])
    _load_linear(model.fc2, d["fc2"])
    _load_linear(model.actor_head, d["actor_head"])
    _load_linear(model.local_value_head, d["local_value_head"])

    return d


def _load_model_weights_binary(model: PolicyNetV2Torch, path: str) -> None:
    data = np.fromfile(path, dtype=np.float32)
    offset = 0
    for _, param in model.named_parameters():
        n = param.numel()
        param.data.copy_(
            torch.from_numpy(data[offset : offset + n]).view(param.shape)
        )
        offset += n


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_linear(linear: nn.Linear, relu: bool) -> dict:
    w = linear.weight.data.cpu().flatten().tolist()
    b = linear.bias.data.cpu().tolist()
    return {
        "weights": w,
        "biases": b,
        "in_dim": linear.in_features,
        "out_dim": linear.out_features,
        "relu": relu,
    }


def _save_layer_norm(ln: nn.LayerNorm) -> dict:
    return {
        "gamma": ln.weight.data.cpu().tolist(),
        "beta": ln.bias.data.cpu().tolist(),
        "dim": ln.normalized_shape[0],
    }


def _save_attention(attn: EntityAttention) -> dict:
    return {
        "w_q": _save_linear(attn.w_q, relu=False),
        "w_k": _save_linear(attn.w_k, relu=False),
        "w_v": _save_linear(attn.w_v, relu=False),
        "w_o": _save_linear(attn.w_o, relu=False),
        "layer_norm": _save_layer_norm(attn.layer_norm),
        "n_heads": attn.n_heads,
        "head_dim": attn.head_dim,
        "embed_dim": attn.embed_dim,
    }


def save_model_weights(model: PolicyNetV2Torch, output_path: str) -> None:
    """Save model weights to Rust JSON or flat binary format."""
    if output_path.endswith(".bin"):
        _save_model_weights_binary(model, output_path)
        return
    d = {
        "ego_encoder": _save_linear(model.ego_encoder, relu=True),
        "entity_encoder": _save_linear(model.entity_encoder, relu=True),
        "attn": _save_attention(model.attn),
        "attn2": _save_attention(model.attn2),
        "fc1": _save_linear(model.fc1, relu=True),
        "fc2": _save_linear(model.fc2, relu=True),
        "actor_head": _save_linear(model.actor_head, relu=False),
        "local_value_head": _save_linear(
            model.local_value_head, relu=False
        ),
        "embed_dim": model.embed_dim,
    }
    with open(output_path, "w") as f:
        json.dump(d, f)


def _save_model_weights_binary(model: PolicyNetV2Torch, path: str) -> None:
    params = []
    for _, param in model.named_parameters():
        params.append(param.data.cpu().flatten().numpy())
    np.concatenate(params).tofile(path)


# ---------------------------------------------------------------------------
# Normalizer persistence
# ---------------------------------------------------------------------------

def save_normalizers(
    path: str, ego_norm: RunningMeanStd, value_norm: ValueNormalizer
) -> None:
    """Save normalizer state alongside model checkpoint."""
    norm_path = path.replace(".json", "_normalizers.json")
    data = {
        "ego": ego_norm.state_dict(),
        "value": {
            "mean": value_norm.mean,
            "var": value_norm.var,
            "count": value_norm.count,
        },
    }
    with open(norm_path, "w") as f:
        json.dump(data, f)


def load_normalizers(
    path: str, ego_norm: RunningMeanStd, value_norm: ValueNormalizer
) -> bool:
    """Load normalizer state from companion file. Returns True if loaded."""
    norm_path = path.replace(".json", "_normalizers.json")
    if not os.path.exists(norm_path):
        return False
    with open(norm_path, "r") as f:
        data = json.load(f)
    if "ego" in data:
        ego_norm.load_state_dict(data["ego"])
    if "value" in data:
        value_norm.mean = data["value"]["mean"]
        value_norm.var = data["value"]["var"]
        value_norm.count = data["value"]["count"]
    return True
