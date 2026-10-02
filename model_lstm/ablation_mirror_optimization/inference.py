"""Portable deployment loader. Requires numpy, torch and adjacent saved artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch


class DeploymentModel:
    def __init__(self, run_dir, device='cpu'):
        root = Path(run_dir)
        self.config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        stats_path = root / 'norm_stats.npz'
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != self.config['normalization_sha256']:
            raise ValueError('Normalization file does not match this deployment')
        with np.load(stats_path, allow_pickle=False) as stats:
            self.mean = np.concatenate([stats[f'{k}_mean'] for k in self.config['x_keys']])
            std = np.concatenate([stats[f'{k}_std'] for k in self.config['x_keys']])
        self.std = np.where(std < 1e-6, 1., std)
        self.device = torch.device(device)
        spec = importlib.util.spec_from_file_location('deployment_network', root / 'model_definition.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        c = self.config
        self.model = module.AssistLSTM(c['input_dim'], c['hidden_layer'], 8,
                                     dropout=c['dropout'], num_layers=c['num_layers'], num_mistakes=2).to(self.device)
        self.model.load_state_dict(torch.load(root / 'model_weights.pth', map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(self, raw_features):
        """Raw ordered features [160,89] or [batch,160,89]; no prior normalization."""
        values = np.asarray(raw_features, dtype=np.float32)
        if values.ndim == 2:
            values = values[None, ...]
        expected = (self.config['window_size'], self.config['input_dim'])
        if values.ndim != 3 or values.shape[1:] != expected or not np.isfinite(values).all():
            raise ValueError(f'Expected finite raw features [batch,{expected[0]},{expected[1]}]')
        x = torch.from_numpy((values - self.mean) / self.std).to(self.device)
        with torch.inference_mode():
            step, progress, mistake = self.model(x)
            result = dict(step_logits=step, step_probabilities=step.softmax(1),
                          step_id=step.argmax(1), progress=progress, progress_percent=progress * 100,
                          mistake_probabilities=mistake.softmax(1), mistake_id=mistake.argmax(1))
        return {k: v.cpu().numpy() for k, v in result.items()}
