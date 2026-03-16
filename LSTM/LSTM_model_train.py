import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# ============================================================
# Model
# ============================================================
class AssistLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_types=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.type_head = nn.Linear(hidden_dim, num_types)
        self.urgency_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, W, F]
        """
        out, _ = self.lstm(x)         # [B, W, H]
        h_last = out[:, -1, :]        # [B, H]

        type_logits = self.type_head(h_last)              # [B, num_types]
        urgency = torch.sigmoid(self.urgency_head(h_last))  # [B, 1]
        urgency = urgency.squeeze(1)                      # [B]

        return type_logits, urgency
