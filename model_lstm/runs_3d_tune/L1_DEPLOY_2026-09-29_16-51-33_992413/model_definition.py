"""LSTM: soft task targets, per-task progress, binary mistake and idle."""

import torch
import torch.nn as nn


class AssistLSTM(nn.Module):
    def __init__(self, input_dim=251, hidden_dim=128, num_steps=7, dropout=0.3, num_layers=1):
        super().__init__()
        if num_steps != 7:
            raise ValueError("This experiment uses 7 task outputs plus a separate idle head")
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
        )
        self.step_head = nn.Linear(hidden_dim, num_steps)
        self.progress_head = nn.Linear(hidden_dim, num_steps)
        self.mistake_head = nn.Linear(hidden_dim, 1)
        self.idle_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        feat = self.shared(h_n[-1])
        return (
            self.step_head(feat), self.progress_head(feat),
            self.mistake_head(feat).squeeze(-1), self.idle_head(feat).squeeze(-1),
        )


class AssistLoss(nn.Module):
    def __init__(self, task_pos_weight, mistake_pos_weight,
                 lambda_step=1.0, lambda_progress=0.3, lambda_mistake=0.3, lambda_idle=0.2):
        super().__init__()
        task_weight = torch.as_tensor(task_pos_weight, dtype=torch.float32)
        mistake_weight = torch.as_tensor(mistake_pos_weight, dtype=torch.float32)
        if task_weight.shape != (7,) or mistake_weight.ndim != 0:
            raise ValueError("Expected 7 task weights and one scalar mistake weight")
        if (not torch.isfinite(task_weight).all() or not torch.isfinite(mistake_weight)
                or (task_weight < 0).any() or mistake_weight < 0):
            raise ValueError("pos_weight must be finite and nonnegative")
        self.step_bce = nn.BCEWithLogitsLoss(pos_weight=task_weight)
        self.mistake_bce = nn.BCEWithLogitsLoss(pos_weight=mistake_weight)
        self.idle_bce = nn.BCEWithLogitsLoss()
        self.weights = dict(step=lambda_step, progress=lambda_progress,
                            mistake=lambda_mistake, idle=lambda_idle)

    def forward(self, outputs, targets):
        step, progress, mistake, idle = outputs
        active = targets["plateau"] >= 0.5
        # An empty mask produces differentiable zero, without dividing by zero.
        progress_loss = (progress - targets["progress"]).square()[active].sum() / active.sum().clamp_min(1)
        parts = {
            "step": self.step_bce(step, targets["step"]),  # includes idle windows
            "progress": progress_loss,
            "mistake": self.mistake_bce(mistake, targets["mistake"]),
            "idle": self.idle_bce(idle, targets["idle"]),
        }
        return sum(self.weights[name] * value for name, value in parts.items()), parts
