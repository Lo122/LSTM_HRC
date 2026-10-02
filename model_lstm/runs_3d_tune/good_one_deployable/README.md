# Final L1 deployment model

Trained from scratch on all 15 subjects, originals + one mirror, for 30 epochs.
Load model_weights.pth with AssistLSTM from model_definition.py and the architecture in config.json.
Apply the feature transforms in config.json, then normalize each panel with norm_stats.npz; concatenate in feature_keys order, preserving the source PT column order within each panel.
Use 120 consecutive frames at 30 fps; outputs describe the last frame. Window hop 10 is the training stride.
The forward result is (step_logits, progress, mistake_logit, idle_logit). step_logits already contains all 7 raw task scores, with shape [batch, 7], in config.json task_names order. No argmax is applied inside the model. Use sigmoid (not softmax) for this soft-peak BCE model; these independent response scores need not sum to one and are not calibrated confidence probabilities. Keep the idle score separate for post-processing. Class 7 is idle.

## Load and return scores for post-processing

Run this example from the deployment directory. The input x must already use this model's transforms, norm and feature order described above.

```python
import json
import torch
from model_definition import AssistLSTM

with open('config.json', encoding='utf-8') as stream:
    config = json.load(stream)
model = AssistLSTM(**{k: config[k] for k in
    ('input_dim', 'hidden_dim', 'num_steps', 'dropout', 'num_layers')})
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu', weights_only=True))
model.eval()

@torch.inference_mode()
def predict(x):
    # x: normalized float32 tensor [batch, 120, 251], on CPU.
    step_logits, progress, mistake_logit, idle_logit = model(x)
    return {
        'step_logits': step_logits,        # [batch, 7], unrestricted real values
        'step_scores': step_logits.sigmoid(),  # [batch, 7], independent [0, 1] scores
        'idle_score': idle_logit.sigmoid(),
        'mistake_score': mistake_logit.sigmoid(),
        'progress': progress,
    }
```

metrics.csv contains in-training metrics with dropout, not an independent test result or a final-checkpoint evaluation. No best checkpoint was selected on these metrics. Keep the original LOSO results for reporting generalization.
