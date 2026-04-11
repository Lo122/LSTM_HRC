# model_rnn.py
# GRU/LSTM baselines for Step-Conditioned Proactive Assistance
#
# Inputs (from dataset.npz produced by build_dataset.py):
#   X_main: [B, W, pose_dim]
#   X_hist: [B, M, W, pose_dim]   (optional, we flatten to [B, M*W, pose_dim])
#   X_step: [B]
#   X_elem: [B, elem_dim]
#   X_env : [B, env_dim]
#   X_pos : [B, pos_dim]
#
# Outputs:
#   need_logits   : [B, 2]
#   type_logits   : [B, num_types]
#   timing_logits : [B, num_timings]

import torch
import torch.nn as nn


# -----------------------------
# Small helper: MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        if in_dim == 0:
            # allow empty features (meta not used)
            self.net = None
            self.out_dim = out_dim
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
            self.out_dim = out_dim

    def forward(self, x):
        if self.net is None:
            # if in_dim==0, x will be [B,0], return zeros
            return torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        return self.net(x)


# -----------------------------
# Core: RNN Encoder + Fusion
# -----------------------------
class StepConditionedAssistRNN(nn.Module):
    """
    Baseline model using GRU or LSTM.

    Strategy:
      1) Encode pose sequence (optionally including step-local history) with RNN
      2) Fuse with static embeddings (element/env/pos) and step embedding
      3) Multi-head classification
    """

    def __init__(
        self,
        pose_dim: int,
        elem_dim: int,
        env_dim: int,
        pos_dim: int,
        num_steps: int = 64,
        num_types: int = 5,      # match label_video.py mapping
        num_timings: int = 3,    # NOW/SOON/LATER
        rnn_type: str = "gru",   # "gru" or "lstm"
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        bidirectional: bool = False,
        step_emb_dim: int = 32,
        static_emb_dim: int = 64,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
        use_history: bool = True,
    ):
        super().__init__()

        self.use_history = use_history
        self.pose_dim = pose_dim
        self.rnn_hidden = rnn_hidden
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1

        # step id embedding
        self.step_emb = nn.Embedding(num_steps, step_emb_dim)

        # static embeddings
        self.elem_mlp = MLP(elem_dim, static_emb_dim, hidden=128, dropout=dropout)
        self.env_mlp = MLP(env_dim, static_emb_dim, hidden=128, dropout=dropout)
        self.pos_mlp = MLP(pos_dim, static_emb_dim, hidden=128, dropout=dropout)

        # input projection (helps RNN)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        rnn_in = rnn_hidden

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_in,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_in,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        # fusion MLP
        # rnn_out = last hidden (or pooled) + step + static
        fused_in = (rnn_hidden * self.num_dir) + step_emb_dim + (static_emb_dim * 3)

        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # heads
        self.need_head = nn.Linear(fusion_hidden, 2)
        self.type_head = nn.Linear(fusion_hidden, num_types)
        self.timing_head = nn.Linear(fusion_hidden, num_timings)

    def forward(self, X_main, X_hist, X_step, X_elem, X_env, X_pos):
        """
        X_main: [B, W, pose_dim]
        X_hist: [B, M, W, pose_dim]
        """
        if self.use_history and X_hist is not None:
            # flatten step-local history into a longer sequence
            B, M, W, D = X_hist.shape
            hist_seq = X_hist.reshape(B, M * W, D)  # [B, M*W, pose_dim]
            seq = torch.cat([hist_seq, X_main], dim=1)  # [B, (M*W+W), pose_dim]
        else:
            seq = X_main  # [B, W, pose_dim]

        # project pose dims
        seq = self.pose_proj(seq)  # [B, T, rnn_hidden]

        # RNN encode
        out, h = self.rnn(seq)

        # get last hidden
        # GRU: h = [layers*dir, B, H]
        # LSTM: h = (h_n, c_n)
        if isinstance(h, tuple):
            h_n = h[0]
        else:
            h_n = h

        # take last layer
        last = h_n[-self.num_dir:, :, :]  # [dir, B, H]
        last = last.transpose(0, 1).reshape(seq.shape[0], -1)  # [B, H*dir]

        # embeddings
        step_emb = self.step_emb(X_step.clamp(min=0, max=self.step_emb.num_embeddings - 1))
        elem_emb = self.elem_mlp(X_elem)
        env_emb = self.env_mlp(X_env)
        pos_emb = self.pos_mlp(X_pos)

        fused = torch.cat([last, step_emb, elem_emb, env_emb, pos_emb], dim=-1)
        fused = self.fusion(fused)

        need_logits = self.need_head(fused)
        type_logits = self.type_head(fused)
        timing_logits = self.timing_head(fused)

        return need_logits, type_logits, timing_logits