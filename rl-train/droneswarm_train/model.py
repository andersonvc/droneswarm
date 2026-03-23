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

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, E = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.w_q(x).view(B, S, H, D).transpose(1, 2)
        k = self.w_k(x).view(B, S, H, D).transpose(1, 2)
        v = self.w_v(x).view(B, S, H, D).transpose(1, 2)

        # Manual attention (more stable on MPS than F.scaled_dot_product_attention).
        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores = scores.clamp(-50.0, 50.0)  # prevent overflow before softmax

        if mask is not None:
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S] True=padding
            scores = scores.masked_fill(pad_mask, -1e4)  # finite for MPS stability

        attn = F.softmax(scores, dim=-1)
        attn = attn.clamp(min=1e-8)  # prevent exact zeros
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.w_o(out)
        out = self.layer_norm(out + x)

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
        ego_embed = F.relu(self.ego_encoder(ego))
        entity_embed = F.relu(self.entity_encoder(entities))

        S = entities.shape[1]
        indices = torch.arange(S, device=ego.device).unsqueeze(0)
        attn_mask = indices < n_entities.unsqueeze(1)

        attn_out = self.attn(entity_embed, mask=attn_mask)
        attn_out = self.attn2(attn_out, mask=attn_mask)

        fill_mask = ~attn_mask.unsqueeze(-1)
        attn_out_masked = attn_out.masked_fill(fill_mask, float("-inf"))

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
