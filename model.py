"""
model.py — Mimi-to-HuBERT Bridge Module (+ Emotion Classification Head)
=========================================================================
Converts Mimi discrete token streams (B, T, 8) → HuBERT-like features (B, 2T, 1024)
                                              + emotion logits (B, num_emotions)

Architecture:
  1. Multi-codebook token embeddings
  2. ConvTranspose1d temporal upsampling (×2)
  3. Causal Transformer with relative positional encoding
  4. Linear output projection to 1024-dim
  5. [NEW] Emotion classification head with attention pooling → num_emotions logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 128, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.causal = causal
        self.embeddings = nn.Embedding(2 * max_distance + 1, num_heads)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.embeddings(rel).permute(2, 0, 1)
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            bias = bias.masked_fill(mask.unsqueeze(0), float("-inf"))
        return bias


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_relative_pe=True, max_distance=128):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.use_relative_pe = use_relative_pe
        if use_relative_pe:
            self.rel_bias = RelativePositionBias(nhead, max_distance, causal=True)

    def reset_cache(self): pass

    def forward(self, x, use_cache=False, past_kv=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def split_heads(t):
            return t.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v) if use_cache else None
        S = k.size(2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if self.use_relative_pe:
            bias = self.rel_bias(S, x.device)
            attn = attn + bias[:, -T:, :]
        else:
            causal_mask = torch.triu(torch.ones(T, S, device=x.device), diagonal=S-T+1).bool()
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out)), present_kv


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, use_relative_pe):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead, dropout, use_relative_pe)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, use_cache=False, past_kv=None):
        attn_out, present_kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, present_kv


class CausalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_ff, dropout, use_relative_pe, max_seq_len):
        super().__init__()
        self.use_relative_pe = use_relative_pe
        if not use_relative_pe:
            self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_ff, dropout, use_relative_pe)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, use_cache=False, past_kvs=None):
        if not self.use_relative_pe:
            x = self.pos_enc(x)
        present_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, pres = layer(x, use_cache=use_cache, past_kv=pkv)
            present_kvs.append(pres)
        x = self.norm(x)
        return x, present_kvs if use_cache else None


class MultiCodebookEmbedding(nn.Module):
    def __init__(self, num_codebooks=8, vocab_size=2048, embed_dim=256, fusion="sum"):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.fusion = fusion
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)])
        if fusion == "concat":
            self.proj = nn.Linear(embed_dim * num_codebooks, embed_dim)
        self.level_scale = nn.Parameter(torch.ones(num_codebooks))
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, tokens):
        embeds = [emb(tokens[:, :, i]) * self.level_scale[i] for i, emb in enumerate(self.embeddings)]
        if self.fusion == "sum":
            return torch.stack(embeds, dim=0).sum(dim=0)
        elif self.fusion == "concat":
            return self.proj(torch.cat(embeds, dim=-1))
        raise ValueError(f"Unknown fusion: {self.fusion}")


class CausalUpsample(nn.Module):
    def __init__(self, channels, upsample_factor=4):
        super().__init__()
        self.factor = upsample_factor
        self.conv_t = nn.ConvTranspose1d(channels, channels, kernel_size=upsample_factor, stride=upsample_factor)
        self.causal_refine = nn.Conv1d(channels, channels, kernel_size=3, padding=0)
        self.causal_pad = 2
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv_t(x)
        x = self.causal_refine(F.pad(x, (self.causal_pad, 0)))
        return self.act(self.norm(x))


# ──────────────────────────────────────────────────────────────────────────────
# [NEW] Emotion Classification Head with Attention Pooling
# ──────────────────────────────────────────────────────────────────────────────

class EmotionHead(nn.Module):
    """
    Classifies emotion from the shared transformer hidden states.

    Design:
      - Attention-weighted pooling over the time axis (learns which frames
        are most emotionally salient — outperforms mean/max pooling).
      - Two-layer MLP with LayerNorm + GELU + Dropout for regularisation.

    Input:  (B, T, d_model)
    Output: (B, num_emotions)  raw logits
    """

    def __init__(self, d_model: int, num_emotions: int, dropout: float = 0.2, hidden_dim: int = 256):
        super().__init__()
        self.attn_pool_w = nn.Linear(d_model, 1, bias=False)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)
        self.fc1    = nn.Linear(d_model, hidden_dim)
        self.act    = nn.GELU()
        self.drop2  = nn.Dropout(dropout)
        self.fc2    = nn.Linear(hidden_dim, num_emotions)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.attn_pool_w.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (B, T) bool — True = valid frame
        """
        scores = self.attn_pool_w(x).squeeze(-1)           # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                # (B, T)
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, d_model)
        pooled  = self.drop(self.norm(pooled))
        return self.fc2(self.drop2(self.act(self.fc1(pooled))))  # (B, num_emotions)


# ──────────────────────────────────────────────────────────────────────────────
# Original Bridge (backward-compatible — used by existing inference.py)
# ──────────────────────────────────────────────────────────────────────────────

class MimiHuBERTBridge(nn.Module):
    """Original bridge without emotion head (kept for backward compat)."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.embed_dim = m["embed_dim"]
        self.d_model = m["d_model"]
        self.output_dim = m["output_dim"]
        self.upsample_factor = m["upsample_factor"]
        self.embedding = MultiCodebookEmbedding(m["num_codebooks"], m["vocab_size"], m["embed_dim"], m["embed_fusion"])
        self.input_proj = nn.Linear(m["embed_dim"], m["d_model"]) if m["embed_dim"] != m["d_model"] else nn.Identity()
        self.upsample = CausalUpsample(m["d_model"], m["upsample_factor"])
        self.transformer = CausalTransformer(
            m["d_model"], m["nhead"], m["num_layers"], m["dim_feedforward"],
            m["dropout"], m["pos_encoding"] == "relative", m["max_seq_len"],
        )
        self.output_proj = nn.Sequential(
            nn.Linear(m["d_model"], m["d_model"]), nn.GELU(), nn.Linear(m["d_model"], m["output_dim"]),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens, use_cache=False, past_kvs=None):
        x = self.input_proj(self.embedding(tokens))
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        x, present_kvs = self.transformer(x, use_cache=use_cache, past_kvs=past_kvs)
        return self.output_proj(x), present_kvs

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}


# ──────────────────────────────────────────────────────────────────────────────
# [NEW] Multi-Task Bridge: features + emotion (main model for this extension)
# ──────────────────────────────────────────────────────────────────────────────

class MimiHuBERTBridgeWithEmotion(nn.Module):
    """
    Mimi → HuBERT Bridge + Emotion Classification (multi-task).

    Shared transformer hidden states feed two task heads:
      • output_proj  → HuBERT-like features  (B, 2T, output_dim)
      • emotion_head → emotion class logits  (B, num_emotions)

    Config reads:
      model.*         — same as original bridge
      emotion.num_classes  — number of emotion categories (default 7)
      emotion.dropout      — dropout in emotion head      (default 0.2)
      emotion.hidden_dim   — MLP hidden size              (default 256)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        e = cfg.get("emotion", {})

        self.embed_dim       = m["embed_dim"]
        self.d_model         = m["d_model"]
        self.output_dim      = m["output_dim"]
        self.upsample_factor = m["upsample_factor"]

        # Shared encoder
        self.embedding   = MultiCodebookEmbedding(m["num_codebooks"], m["vocab_size"], m["embed_dim"], m["embed_fusion"])
        self.input_proj  = nn.Linear(m["embed_dim"], m["d_model"]) if m["embed_dim"] != m["d_model"] else nn.Identity()
        self.upsample    = CausalUpsample(m["d_model"], m["upsample_factor"])
        self.transformer = CausalTransformer(
            m["d_model"], m["nhead"], m["num_layers"], m["dim_feedforward"],
            m["dropout"], m["pos_encoding"] == "relative", m["max_seq_len"],
        )

        # Task heads
        self.output_proj = nn.Sequential(
            nn.Linear(m["d_model"], m["d_model"]), nn.GELU(), nn.Linear(m["d_model"], m["output_dim"]),
        )
        self.emotion_head = EmotionHead(
            d_model      = m["d_model"],
            num_emotions = e.get("num_classes", 7),
            dropout      = e.get("dropout", 0.2),
            hidden_dim   = e.get("hidden_dim", 256),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kvs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        """
        tokens  : (B, T, num_codebooks)
        mask    : (B, 2T) bool — True = valid upsampled frame
        returns : features (B,2T,output_dim), emotion_logits (B,num_emotions), kvs
        """
        x = self.input_proj(self.embedding(tokens))          # (B, T, d_model)
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2) # (B, 2T, d_model)
        hidden, present_kvs = self.transformer(x, use_cache=use_cache, past_kvs=past_kvs)

        features       = self.output_proj(hidden)            # (B, 2T, output_dim)
        emotion_logits = self.emotion_head(hidden, mask)     # (B, num_emotions)

        return features, emotion_logits, present_kvs

    def get_param_count(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bridge_p  = sum(p.numel() for name, p in self.named_parameters() if "emotion_head" not in name)
        emotion_p = sum(p.numel() for p in self.emotion_head.parameters())
        return {"total": total, "trainable": trainable, "bridge": bridge_p, "emotion_head": emotion_p}


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureDiscriminator(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, num_layers=4):
        super().__init__()
        layers, in_ch = [], input_dim
        for _ in range(num_layers):
            layers += [nn.Conv1d(in_ch, hidden, 4, 2, 1), nn.LeakyReLU(0.2, True), nn.Dropout(0.1)]
            in_ch = hidden
        layers.append(nn.Conv1d(in_ch, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2))
