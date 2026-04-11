# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Positional Encoding (standard)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:, :L, :]


# -----------------------------
# Feed Forward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Transformer Encoder Block
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [B, L, d_model]
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# -----------------------------
# Simple MLP Encoder for static features
# -----------------------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Step-Conditioned Assist Transformer (Encoder-only)
# ============================================================
class StepConditionedAssistTransformer(nn.Module):
    """
    Token layout:
      [STEP] [ELEMENT] [ENV] [POS] [POSE_1 ... POSE_L]

    Pose tokens come from step-local history:
      X_hist: [B, M, T, pose_dim]
      X_cur:  [B, T, pose_dim]
      -> pose tokens length = (M+1)*T
    """

    def __init__(
        self,
        pose_dim: int,
        step_vocab: int,
        element_dim: int,
        env_dim: int,
        pos_dim: int,
        num_types: int,
        num_timing: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_tokens: int = 4096
    ):
        super().__init__()

        self.d_model = d_model

        # Pose -> token embedding
        self.pose_proj = nn.Linear(pose_dim, d_model)

        # Conditioning encoders
        self.step_emb = nn.Embedding(step_vocab, d_model)
        self.element_enc = MLPEncoder(element_dim, d_model, dropout=dropout) if element_dim > 0 else None
        self.env_enc = MLPEncoder(env_dim, d_model, dropout=dropout) if env_dim > 0 else None
        self.pos_enc = MLPEncoder(pos_dim, d_model, dropout=dropout) if pos_dim > 0 else None

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_tokens)

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.head_need = nn.Linear(d_model, 2)
        self.head_type = nn.Linear(d_model, num_types)
        self.head_timing = nn.Linear(d_model, num_timing)

    def forward(self, X_cur, X_hist, step_id, element_feat=None, env_feat=None, pos_feat=None):
        """
        X_cur:  [B, T, pose_dim]
        X_hist: [B, M, T, pose_dim]  (M can be 0)
        step_id: [B]
        element_feat: [B, element_dim]
        env_feat: [B, env_dim]
        pos_feat: [B, pos_dim]
        """

        B, T, pose_dim = X_cur.shape

        # --- concat pose windows ---
        if X_hist is None:
            X_all = X_cur
        else:
            # [B, M*T, pose_dim]
            X_hist_flat = X_hist.reshape(B, -1, pose_dim)
            X_all = torch.cat([X_hist_flat, X_cur], dim=1)  # [B, L_pose, pose_dim]

        # --- pose tokens ---
        pose_tokens = self.pose_proj(X_all)  # [B, L_pose, d_model]
        pose_tokens = self.positional_encoding(pose_tokens)

        # --- conditioning tokens ---
        step_token = self.step_emb(step_id).unsqueeze(1)  # [B, 1, d_model]

        tokens = [step_token]

        if self.element_enc is not None:
            assert element_feat is not None
            tokens.append(self.element_enc(element_feat).unsqueeze(1))
        else:
            tokens.append(torch.zeros(B, 1, self.d_model, device=X_cur.device))

        if self.env_enc is not None:
            assert env_feat is not None
            tokens.append(self.env_enc(env_feat).unsqueeze(1))
        else:
            tokens.append(torch.zeros(B, 1, self.d_model, device=X_cur.device))

        if self.pos_enc is not None:
            assert pos_feat is not None
            tokens.append(self.pos_enc(pos_feat).unsqueeze(1))
        else:
            tokens.append(torch.zeros(B, 1, self.d_model, device=X_cur.device))

        # final input token sequence
        X = torch.cat(tokens + [pose_tokens], dim=1)  # [B, L_total, d_model]

        # --- transformer encoder ---
        for layer in self.layers:
            X = layer(X)

        # Use STEP token output as summary
        h = X[:, 0, :]  # [B, d_model]

        logits_need = self.head_need(h)
        logits_type = self.head_type(h)
        logits_timing = self.head_timing(h)

        return logits_need, logits_type, logits_timing
