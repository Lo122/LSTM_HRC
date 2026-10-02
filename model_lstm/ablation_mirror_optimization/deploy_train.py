"""Train all 15 participants with the successful mirror optimization settings.

python -B -m model_lstm.ablation_mirror_optimization.deploy_train
No held-out selection: train for the fixed LOSO budget and save the final model.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from model_lstm.LSTM_deploy_train import full_training_files
from model_lstm.loso_initial_mirror.data import (
    SETTINGS, FEATURE_KEYS, RawRecording, WindowDataset, fit_norm, training_weights, source_identity,
)
from model_lstm.loso_initial_mirror.model import MirrorLSTM
from model_lstm.ablation_mirror_optimization.engine import OPTIMIZATION, run_epoch
from model_lstm.ablation_mirror_optimization.run_loso import DEFAULT_DATA, write_json
from model_lstm.ablation_step_supervision.report import save_confusion
from data_proc_2d.app.build_norm_dataset_tune import normalization_id

DEFAULT_LOSO = Path(__file__).parent / 'runs/MIRROR_L1OPT_2026-10-01_17-14-28_539317'


def save_weights(model, path):
    # Export the original network only; loss class weights are saved in config.
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != 'class_weights'}
    temporary = path.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)


def train_full_dataset(data_root, output_dir, loso_run=DEFAULT_LOSO, epochs=None, seed=42,
                       device=None, cache_dir=None):
    data_root, output_dir, loso_run = (Path(p).resolve() for p in (data_root, output_dir, loso_run))
    if output_dir.exists():
        raise ValueError('Choose a new output directory; deployment results are not overwritten')
    reference = json.loads((loso_run / 'protocol.json').read_text(encoding='utf-8'))
    expected = dict(SETTINGS)
    expected.update(OPTIMIZATION)
    if reference.get('experiment') != 'mirror_l1_optimization_v1' or any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError('LOSO reference differs from the configured model/optimization settings')
    epochs = reference['epochs'] if epochs is None else epochs
    if epochs < 1:
        raise ValueError('epochs must be positive')
    paths = full_training_files(data_root)
    cache_dir = Path(cache_dir) if cache_dir else ROOT / 'model_lstm/.dataset_cache/initial_mirror_raw'
    print(f'Preparing all 15 participants: {len(paths)} files, {epochs} epochs', flush=True)
    recordings = [RawRecording(p, cache_dir) for p in paths]
    dimensions = recordings[0].dimensions
    if sum(dimensions) != SETTINGS['input_dim'] or any(r.dimensions != dimensions for r in recordings):
        raise ValueError('Feature layout does not match the model')
    # New full-training norm; never reuse a fold norm or fit on sampled batches.
    stats = fit_norm(recordings)
    dataset = ConcatDataset([WindowDataset(r, stats) for r in recordings])
    sampler, counts, weights = training_weights(dataset, seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_dir.mkdir(parents=True)
    np.savez(output_dir / 'norm_stats.npz', **stats)
    digest = hashlib.sha256((output_dir / 'norm_stats.npz').read_bytes()).hexdigest()
    manifest = dict(profile='initial_mirror_raw_v1', feature_keys=FEATURE_KEYS,
                    subjects=dict(train=list(range(1, 16)), val=[], test=[]),
                    sources=source_identity(dict(train=paths)), dimensions=dimensions,
                    normalization_id=normalization_id(stats), normalization_sha256=digest,
                    statistics_frame_step=1, statistics_dtype='float32', std_ddof=0)
    write_json(output_dir / 'dataset_manifest.json', manifest)
    layout, start = [], 0
    for key, width in zip(FEATURE_KEYS, dimensions):
        layout.append(dict(feature=key, start=start, stop=start + width))
        start += width
    config = dict(expected, epochs=epochs, seed=seed, device=str(device),
                  stage='full_dataset_mirror_l1_optimization', reference_loso=str(loso_run),
                  train_uids=list(range(1, 16)), val_uids=[], test_uids=[],
                  checkpoint_selection='final_epoch_fixed_before_training',
                  epoch_selection='LOSO training budget' if epochs == reference['epochs'] else 'explicit override',
                  scheduler_t_max=epochs, scheduler_eta_min=0.,
                  sampler='WeightedRandomSampler', sampler_weighting='1/sqrt(window_class_count)',
                  sampler_replacement=True, sampler_num_samples=len(dataset),
                  step_class_counts=counts, step_class_weights=weights, step_loss_reduction='weighted_sample_mean',
                  normalization_id=manifest['normalization_id'], normalization_sha256=digest,
                  normalization_file='norm_stats.npz', model_file='model_weights.pth',
                  feature_layout=layout, feature_transform='none', input_fps=30,
                  output_classes=list(range(8)), background_class=7,
                  progress_scaling='raw target divided by 100; model output linear and not clipped')
    write_json(output_dir / 'config.json', config)
    write_json(output_dir / 'reference_loso_protocol.json', reference)
    source = output_dir / 'source'
    source.mkdir()
    for path in (Path(__file__), Path(__file__).with_name('engine.py')):
        shutil.copy2(path, source / path.name)
    for relative in ('model_lstm/loso_initial_mirror/data.py', 'model_lstm/loso_initial_mirror/model.py',
                     'model_lstm/loso_initial_mirror/reference_config.json',
                     'model_lstm/LSTM_deploy_train.py', 'model_lstm/LSTM_model_train.py',
                     'model_lstm/ablation_step_supervision/engine.py',
                     'model_lstm/ablation_step_supervision/report.py',
                     'model_lstm/ablation_step_supervision/data.py',
                     'model_lstm/LSTM_databuilder_tune.py',
                     'model_lstm/ablation_mirror_optimization/run_loso.py',
                     'data_proc_2d/app/build_norm_dataset_tune.py'):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(ROOT / 'model_lstm/LSTM_model_train.py', output_dir / 'model_definition.py')
    shutil.copy2(Path(__file__).with_name('inference.py'), output_dir / 'inference.py')
    (output_dir / 'README.md').write_text(
        '# Full-data mirror optimization deployment\n\n'
        f'Trained from scratch on all 15 participants, originals plus one mirror, for {epochs} epochs.\n'
        'No validation/test selection; model_weights.pth is the final epoch. '
        'Temporary last-epoch weights are kept during training and removed on successful completion.\n\n'
        'Keep config.json, norm_stats.npz, model_weights.pth, model_definition.py and inference.py together.\n'
        'Requires Python, numpy and torch. Input: 160 consecutive frames at 30 fps, 89 raw features. '
        'Use the exact feature order in config.json feature_layout; do not normalize beforehand. '
        'Training hop=30 is not a mandatory inference update interval. Output refers to the last frame.\n\n'
        '```python\nfrom inference import DeploymentModel\n'
        'predictor = DeploymentModel("path/to/this/directory", device="cpu")\n'
        'prediction = predictor.predict(raw_features)  # [160,89] or [batch,160,89]\n```\n\n'
        'step_logits and step_probabilities each have 8 columns: steps 0..6, background 7. '
        'Probabilities use softmax, not independent sigmoid lanes. '
        'Progress is linear and not clipped; mistake has 2 softmax probabilities.\n\n'
        'metrics.csv contains sampled training metrics; train_results.json and confusion plots '
        'evaluate every training window once. These are not held-out generalization results. '
        'Report the separate LOSO experiment for generalization.\n', encoding='utf-8')
    loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], sampler=sampler, shuffle=False,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=device.type == 'cuda', drop_last=False)
    torch.manual_seed(seed)
    model = MirrorLSTM(SETTINGS['input_dim'], weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZATION['lr'],
                                 weight_decay=OPTIMIZATION['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        metrics, _, _ = run_epoch(model, loader, device, optimizer,
                                  gradient_clip=OPTIMIZATION['gradient_clip'])
        row = dict(epoch=epoch, learning_rate=lr, **{f'train_{k}': v for k, v in metrics.items()})
        with (output_dir / 'metrics.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        scheduler.step()
        save_weights(model, output_dir / 'last_model_weights.pth')
        write_json(output_dir / 'last_checkpoint.json', dict(completed_epochs=epoch, planned_epochs=epochs))
        print(f"Epoch {epoch}/{epochs}: lr={lr:.6g}, sampled train F1={metrics['macro_f1_8']:.4f}", flush=True)
    save_weights(model, output_dir / 'model_weights.pth')
    eval_loader = DataLoader(dataset, batch_size=SETTINGS['batch_size'], shuffle=False, num_workers=0)
    metrics, cm, _ = run_epoch(model, eval_loader, device)
    write_json(output_dir / 'train_results.json', metrics)
    save_confusion(cm, output_dir, 'Full training set, final checkpoint, no sampler (not held-out)')
    write_json(output_dir / 'training_summary.json', dict(completed_epochs=epochs,
        source_files=len(paths), training_windows=len(dataset), checkpoint='model_weights.pth',
        selection='final_epoch_fixed_before_training',
        evaluation='Training diagnostics only. Use the LOSO results to report generalization.'))
    (output_dir / 'last_model_weights.pth').unlink()
    (output_dir / 'last_checkpoint.json').unlink()
    print(f'Deployment saved: {output_dir}', flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--loso-run', type=Path, default=DEFAULT_LOSO)
    parser.add_argument('--epochs', type=int, help='Default: reference LOSO budget (12)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(__file__).parent / 'runs' / datetime.now().strftime('MIRROR_DEPLOY_%Y-%m-%d_%H-%M-%S_%f')
    train_full_dataset(args.data_root, output, args.loso_run, args.epochs, args.seed, args.device, args.cache_dir)


if __name__ == '__main__':
    main()
