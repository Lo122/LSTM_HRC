"""Three step-supervision designs with the same backbone and auxiliary tasks."""

import torch
from torch import nn
from torch.nn import functional as F


class SharedLSTM(nn.Module):
    def __init__(self, input_dim=91, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        # Construct common modules before arm-specific heads so equal seeds give
        # identical backbone AND auxiliary initialization across all three arms.
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.mistake_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        features = self.shared(hidden[-1])
        return dict(
            step=self.step_head(features), progress=self.progress_head(features).squeeze(-1),
            mistake=self.mistake_head(features),
            background=self.background_head(features).squeeze(-1) if hasattr(self, "background_head") else None,
        )

    def losses(self, out, target):
        step, background, step_count = self.step_losses(out, target)
        parts = dict(step=step, progress=F.mse_loss(out["progress"], target["progress"]),
                     mistake=F.cross_entropy(out["mistake"], target["mistake"]), background=background)
        # These are fixed multitask coefficients, not class/sample weights.
        total = parts["step"] + parts["progress"] + parts["mistake"] + 0.2 * parts["background"]
        return total, parts, step_count


class AIntegerCE(SharedLSTM):
    """A: eight-way CE, including background as class 7."""
    def __init__(self, input_dim=91, hidden_dim=64, dropout=0.2):
        super().__init__(input_dim, hidden_dim, dropout)
        self.step_head = nn.Linear(hidden_dim, 8)

    def step_losses(self, out, target):
        return F.cross_entropy(out["step"], target["step"]), out["step"].sum() * 0, len(target["step"])

    def decode(self, out):
        return out["step"].argmax(1), out["step"].softmax(1)[:, 7]


class BSeparatedCE(SharedLSTM):
    """B: seven-way CE on non-background windows and a separate binary head."""
    def __init__(self, input_dim=91, hidden_dim=64, dropout=0.2):
        super().__init__(input_dim, hidden_dim, dropout)
        self.step_head = nn.Linear(hidden_dim, 7)
        self.background_head = nn.Linear(hidden_dim, 1)

    def task_loss(self, logits, target, active):
        # Clamping only supplies a valid placeholder for masked background rows.
        per_window = F.cross_entropy(logits, target["step"].clamp_max(6), reduction="none")
        return (per_window * active).sum() / active.sum().clamp_min(1)

    def step_losses(self, out, target):
        active = target["step"] != 7
        background = F.binary_cross_entropy_with_logits(out["background"], (~active).float())
        return self.task_loss(out["step"], target, active), background, active.sum()

    def decode(self, out):
        score = out["background"].sigmoid()
        prediction = torch.where(score > 0.5, 7, out["step"].argmax(1))
        return prediction, score


class CSoftPeak(BSeparatedCE):
    """C: soft-peak BCE, using exactly B's non-background step-loss mask."""
    def task_loss(self, logits, target, active):
        per_window = F.binary_cross_entropy_with_logits(logits, target["peak"], reduction="none").mean(1)
        return (per_window * active).sum() / active.sum().clamp_min(1)


MODELS = {"A": AIntegerCE, "B": BSeparatedCE, "C": CSoftPeak}
