"""ATSCC block order, ITSC scaled backbone, and plan-defined output dimensions.

Reference: petchthwr/ATSCC, commit 789ccd72b7496b52e03df7d8599d8c9e6659bdb9.
Preserves upstream projection L2 norms, bias-free attention, FFN dropout and
normalization/residual order. No extra final LayerNorm or positional encoding.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Attention(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.dim, self.heads, self.dropout = model_dim, heads, dropout
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x, allowed):
        b, t, d = x.shape
        def heads(projection):
            return projection(x).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = [heads(p) for p in (self.q_proj, self.k_proj, self.v_proj)]
        # The upstream implementation scales by sqrt(model_dim), not sqrt(head_dim).
        # SDPA gives zero attention for a fully masked causal prefix, avoiding
        # upstream's uniform softmax over disallowed future keys in that case.
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                    dropout_p=self.dropout if self.training else 0.0,
                    scale=1.0 / math.sqrt(self.dim))
        result = result.transpose(1, 2).reshape(b, t, d)
        return F.dropout(self.out_proj(result), p=self.dropout, training=self.training)


class Block(nn.Module):
    def __init__(self, model_dim, ffn_dim, heads, dropout):
        super().__init__()
        self.attn_norm, self.ff_norm = nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)
        self.attn = Attention(model_dim, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(model_dim, ffn_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ffn_dim, model_dim))

    def forward(self, x, allowed):
        x = self.attn_norm(x)
        x = x + self.attn(x, allowed)
        x = self.ff_norm(x)
        return x + self.ff(x)


class TrajectoryEncoder(nn.Module):
    def __init__(self, output_dim, input_dim=9, model_dim=256, ffn_dim=512,
                 heads=4, layers=4, dropout=.35, mask_probability=.2):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.mask_probability = mask_probability
        self.pre_proj = nn.Linear(input_dim, model_dim)
        self.layers = nn.ModuleList([Block(model_dim, ffn_dim, heads, dropout) for _ in range(layers)])
        self.post_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x, valid, input_mask=None):
        if input_mask is None:
            input_mask = (torch.rand(valid.shape, device=x.device) < self.mask_probability
                          if self.training else torch.zeros_like(valid))
        hidden = input_mask | ~valid
        h = F.normalize(self.pre_proj(x.masked_fill(hidden[..., None], 0)).float(), dim=-1)
        causal = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & (~hidden[:, None, None, :])
        for layer in self.layers:
            h = layer(h, allowed)
        z = F.normalize(self.post_proj(h).float(), dim=-1)
        # Input token masking is an augmentation. As in ATSCC, all non-padding
        # timesteps are supervised by their RDP identity, including masked inputs.
        return z.masked_fill(~valid[..., None], 0)


def flat_snn(z, labels, valid, temperature, full_limit=7000, chunk_size=512):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    trajectory = torch.arange(z.shape[0], device=z.device)[:, None].expand_as(labels)
    segment = trajectory[valid] * (z.shape[1] + 1) + labels[valid]
    emb = F.normalize(z[valid].float(), dim=-1)
    counts = torch.bincount(segment)
    usable = (counts[segment] > 1) & (counts[segment] < len(segment))
    anchors = usable.nonzero(as_tuple=True)[0]
    if not len(anchors):
        raise ValueError("No segment anchors with both positive and negative partners")
    def block(x, anchor, identities):
        score = x[anchor] @ x.T / temperature
        same = identities[anchor, None] == identities[None, :]
        positive = same & (anchor[:, None] != torch.arange(len(x), device=x.device)[None, :])
        return (torch.logsumexp(score.masked_fill(same, -torch.inf), 1)
                - torch.logsumexp(score.masked_fill(~positive, -torch.inf), 1)).sum()
    with torch.autocast(device_type=z.device.type, enabled=False):
        if len(emb) <= full_limit:
            loss = block(emb, anchors, segment)
        else:
            loss = emb.sum() * 0
            for part in anchors.split(chunk_size):
                loss = loss + (checkpoint(block, emb, part, segment, use_reentrant=False)
                               if torch.is_grad_enabled() and emb.requires_grad else block(emb, part, segment))
    return loss / len(anchors), len(anchors)


def pool(tokens, mode):
    if mode == "sum":
        return tokens.sum(0)
    if mode == "mean":
        return tokens.mean(0)
    if mode == "gem2":
        # ITSC equation on page 3: signed generalized mean, p=2.
        value = (tokens.sign() * tokens.abs().square()).mean(0)
        return value.sign() * value.abs().sqrt()
    raise ValueError(mode)
