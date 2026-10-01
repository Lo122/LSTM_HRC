# Final L1 deployment model

Trained from scratch on all 15 subjects, originals + one mirror, for 9 epochs.
Load model_weights.pth with AssistLSTM from model_definition.py and the architecture in config.json.
Apply the feature transforms in config.json, then normalize each panel with norm_stats.npz; concatenate in feature_keys order, preserving the source PT column order within each panel.
Use 120 consecutive frames at 30 fps; outputs describe the last frame. Window hop 10 is the training stride.
The forward result is (step_logits, progress, mistake_logit, idle_logit). Use config.json decoding and task_names; class 7 is idle.

metrics.csv contains in-training metrics with dropout, not an independent test result or a final-checkpoint evaluation. No best checkpoint was selected on these metrics. Keep the original LOSO results for reporting generalization.
