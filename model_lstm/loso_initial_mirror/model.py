"""Original notebook network and weighted sample-mean CE objective."""
import torch
from torch.nn import functional as F
from model_lstm.LSTM_model_train import AssistLSTM
from .data import SETTINGS


class MirrorLSTM(AssistLSTM):
    def __init__(self, input_dim, class_weights):
        super().__init__(input_dim, SETTINGS['hidden_layer'], 8, dropout=SETTINGS['dropout'],
                         num_layers=SETTINGS['num_layers'], num_mistakes=2)
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, x):
        step, progress, mistake = super().forward(x)
        return dict(step=step, progress=progress, mistake=mistake, background=None)

    def losses(self, out, y):
        parts = dict(step=F.cross_entropy(out['step'], y['step'], weight=self.class_weights, reduction='none').mean(),
                     progress=F.mse_loss(out['progress'], y['progress']),
                     mistake=F.cross_entropy(out['mistake'], y['mistake']),
                     background=out['step'].sum() * 0)
        return sum(parts[k] * SETTINGS[f'lambda_{k}'] for k in ('step', 'progress', 'mistake')), parts, len(y['step'])

    def decode(self, out):
        return out['step'].argmax(1), out['step'].softmax(1)[:, 7]
