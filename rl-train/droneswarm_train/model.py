"""Entity-attention Actor-Critic policy network (PyTorch).

Architecture matches the Rust PolicyNetV2 / WASM InferenceNetV2 exactly:
  ego -> ego_encoder(Linear+ReLU) -> [64]
  entities -> entity_encoder(Linear+ReLU) -> [S, 64]
  -> attn -> attn2 -> max_pool -> [64]
  concat(ego_embed, pooled) -> [128]
  -> fc1(Linear+ReLU) -> fc2(Linear+ReLU) -> [256]
  -> actor_head(Linear) -> [ACT_DIM]
  -> local_value_head(Linear) -> [1]
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ACT_DIM, EGO_DIM, ENTITY_DIM


class EntityAttention(nn.Module):
    """Multi-head self-attention with residual connection and LayerNorm.

    Matches Rust MultiHeadAttention: separate Q/K/V/O linear projections,
    scaled dot-product attention, output projection, residual, LayerNorm.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=True)

        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-4)  # higher eps for MPS stability

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, E = x.shape
        H = self.n_heads
        D = self.head_dim

        # Pre-LN: normalize before attention (more stable gradients than Post-LN).
        x_norm = self.layer_norm(x.float()).to(x.dtype)

        q = self.w_q(x_norm).view(B, S, H, D).transpose(1, 2)
        k = self.w_k(x_norm).view(B, S, H, D).transpose(1, 2)
        v = self.w_v(x_norm).view(B, S, H, D).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
        scores = scores.clamp(-20.0, 20.0)

        if mask is not None:
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, -1e4)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.float())

        out = out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.w_o(out)
        # Residual connection bypasses LayerNorm (clean gradient path).
        out = out + x

        return out


class PolicyNetV2Torch(nn.Module):
    """Entity-attention Actor-Critic policy network."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 64

        self.ego_encoder = nn.Linear(EGO_DIM, 64)
        self.entity_encoder = nn.Linear(ENTITY_DIM, 64)

        self.attn = EntityAttention(64, 4)
        self.attn2 = EntityAttention(64, 4)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)

        self.actor_head = nn.Linear(256, ACT_DIM)
        self.local_value_head = nn.Linear(256, 1)

    def forward(
        self,
        ego: torch.Tensor,
        entities: torch.Tensor,
        n_entities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Force float32 throughout for MPS stability.
        ego = ego.float()
        entities = entities.float()

        ego_embed = F.relu(self.ego_encoder(ego))
        entity_embed = F.relu(self.entity_encoder(entities))

        S = entities.shape[1]
        indices = torch.arange(S, device=ego.device).unsqueeze(0)
        attn_mask = indices < n_entities.unsqueeze(1)

        attn_out = self.attn(entity_embed, mask=attn_mask)
        attn_out = self.attn2(attn_out, mask=attn_mask)

        # Max pool: use large negative instead of -inf to avoid NaN in MPS backward.
        fill_mask = ~attn_mask.unsqueeze(-1)
        attn_out_masked = attn_out.masked_fill(fill_mask, -1e4)

        pooled = attn_out_masked.max(dim=1).values
        all_padded = n_entities == 0
        if all_padded.any():
            pooled = pooled.masked_fill(all_padded.unsqueeze(-1), 0.0)

        trunk_input = torch.cat([ego_embed, pooled], dim=-1)

        h1 = F.relu(self.fc1(trunk_input))
        h2 = F.relu(self.fc2(h1))

        logits = self.actor_head(h2)
        values = self.local_value_head(h2).squeeze(-1)

        return logits, values
