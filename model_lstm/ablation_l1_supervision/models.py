"""Same L1 network and auxiliary losses; only the step objective differs."""

import torch
from torch.nn import functional as F
from model_lstm.LSTM_model_train_tune import AssistLSTM
from model_lstm.LSTM_engine_tune import L1_SETTINGS


class L1HardCE(AssistLSTM):
    def __init__(self, task_pos_weight, mistake_pos_weight):
        super().__init__(**{k: L1_SETTINGS[k] for k in
                           ("input_dim", "hidden_dim", "num_steps", "num_layers", "dropout")})
        self.register_buffer("task_pos_weight", torch.as_tensor(task_pos_weight, dtype=torch.float32))
        self.register_buffer("mistake_pos_weight", torch.as_tensor(mistake_pos_weight, dtype=torch.float32))

    def forward(self, x):
        step, progress, mistake, idle = super().forward(x)
        return dict(step=step, progress=progress, mistake=mistake, background=idle)

    def step_loss(self, logits, target):
        return F.cross_entropy(logits, target["step"].clamp_max(6), reduction="none")

    def losses(self, out, target):
        active = target["step"] != 7
        progress_mask = target["plateau"] >= 0.5
        parts = dict(
            step=(self.step_loss(out["step"], target) * active).sum() / active.sum().clamp_min(1),
            progress=((out["progress"] - target["progress"]).square() * progress_mask).sum()
                     / progress_mask.sum().clamp_min(1),
            mistake=F.binary_cross_entropy_with_logits(out["mistake"], target["mistake"],
                                                       pos_weight=self.mistake_pos_weight),
            background=F.binary_cross_entropy_with_logits(out["background"], target["idle"]),
        )
        weights = L1_SETTINGS["loss_weights"]
        total = sum(parts[k] * weights["idle" if k == "background" else k] for k in parts)
        return total, parts, active.sum()

    def decode(self, out):
        score = out["background"].sigmoid()
        return torch.where(score > 0.5, 7, out["step"].argmax(1)), score


class L1SoftPeak(L1HardCE):
    def step_loss(self, logits, target):
        return F.binary_cross_entropy_with_logits(
            logits, target["peak"], pos_weight=self.task_pos_weight, reduction="none",
        ).mean(1)


MODELS = {"B": L1HardCE, "C": L1SoftPeak}
