"""LOSO for the frozen exp_2026-09-24_17-02-55 reference configuration.

python -B -m model_lstm.loso_initial_mirror.run_loso
Use --output-dir to resume, --folds to select participants.
"""
import argparse
from datetime import datetime
import gc
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
from torch.utils.data import DataLoader
from model_lstm.loso_initial_mirror.data import SETTINGS, load_fold, training_weights, source_identity
from model_lstm.loso_initial_mirror.model import MirrorLSTM
from model_lstm.ablation_step_supervision.engine import fit, run_epoch
from model_lstm.ablation_step_supervision.report import save_confusion, summarize
from data_proc_2d.app.build_norm_dataset_tune import l1_file_splits

DEFAULT_DATA = Path('G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/'
                    'ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_loso(data_root, run_root, folds=tuple(range(1, 16)), epochs=60, seed=42,
             selection='macro_f1', device=None, cache_dir=None):
    if (not folds or len(set(folds)) != len(folds) or set(folds) - set(range(1, 16))
            or epochs < 1 or selection not in ('macro_f1', 'val_loss')):
        raise ValueError('Invalid folds, epochs or selection')
    data_root, run_root = Path(data_root).resolve(), Path(run_root).resolve()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cache_dir = Path(cache_dir) if cache_dir else ROOT / 'model_lstm/.dataset_cache/initial_mirror_raw'
    checkpoint = 'best_macro_f1.pth' if selection == 'macro_f1' else 'best_val_loss.pth'
    protocol = dict(SETTINGS, experiment='initial_mirror_loso_v2', data_root=str(data_root),
                    epochs=epochs, seed=seed, folds=list(folds), device=str(device),
                    selection=selection, test_checkpoint=checkpoint,
                    evaluation='task_id 0..7, complete eight-class macro F1',
                    sampler='WeightedRandomSampler', sampler_weighting='1/sqrt(window_class_count)',
                    step_loss_reduction='weighted_sample_mean', optimizer='Adam',
                    scheduler=None, gradient_clip=None, normalization='raw float32 all-frame mean/std + 1e-6',
                    feature_transform='none; no ratio clipping, no azimuth sin/cos',
                    progress_scaling='raw task_progress / 100 exactly once')
    resuming = run_root.exists()
    if resuming:
        previous = json.loads((run_root / 'protocol.json').read_text())
        differences = [k for k in protocol if k not in ('folds', 'device') and previous.get(k) != protocol[k]]
        if differences:
            raise ValueError(f'Existing experiment settings differ: {differences}')
        # Check all stored folds, even those not requested in this invocation.
        for path in (run_root / 'mirror').glob('fold_*/dataset_manifest.json'):
            manifest = json.loads(path.read_text())
            uid = int(path.parent.name.split('_')[1])
            if source_identity(l1_file_splits(data_root, uid)) != manifest['sources']:
                raise ValueError(f'Source data changed: {path}')
            if hashlib.sha256((path.parent / 'norm_stats.npz').read_bytes()).hexdigest() != manifest['normalization_sha256']:
                raise ValueError(f'Normalization changed: {path}')
        protocol['folds'] = sorted(set(previous['folds']) | set(folds))
    else:
        # Validate source inventory before creating a result directory.
        l1_file_splits(data_root, folds[0])
        run_root.mkdir(parents=True)
        source = run_root / 'source'
        source.mkdir()
        for name in ('run_loso.py', 'data.py', 'model.py', 'reference_config.json', 'README.md'):
            shutil.copy2(Path(__file__).with_name(name), source / name)
        for relative in ('model_lstm/LSTM_model_train.py',
                         'model_lstm/ablation_step_supervision/engine.py',
                         'model_lstm/ablation_step_supervision/report.py',
                         'model_lstm/ablation_step_supervision/data.py',
                         'model_lstm/LSTM_databuilder_tune.py',
                         'data_proc_2d/app/build_norm_dataset_tune.py'):
            destination = source / 'dependencies' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
    write_json(run_root / 'protocol.json', protocol)
    summarize(run_root, ['mirror'])
    for uid in folds:
        directory = run_root / 'mirror' / f'fold_{uid:02d}'
        if (directory / 'test_results.json').exists():
            print(f'Skipping completed UID {uid:02d}', flush=True)
            continue
        print(f'\nInitial mirror LOSO UID {uid:02d}: {epochs} epochs, {device}', flush=True)
        datasets, manifest = load_fold(data_root, directory, uid, cache_dir)
        fold_seed = seed + uid - 1
        sampler, counts, weights = training_weights(datasets['train'], fold_seed)
        config = dict(protocol, test_uid=uid, fold_seed=fold_seed,
                      input_dim=manifest['input_dim'], subjects=manifest['subjects'],
                      normalization_id=manifest['normalization_id'],
                      normalization_sha256=manifest['normalization_sha256'],
                      window_counts={k: len(v) for k, v in datasets.items()},
                      step_class_counts=counts, step_class_weights=weights)
        write_json(directory / 'config.json', config)
        print(f"Input dimensions {manifest['input_dim']}; windows {config['window_counts']}", flush=True)
        (directory / 'metrics.csv').unlink(missing_ok=True)
        loaders = {split: DataLoader(dataset, batch_size=SETTINGS['batch_size'],
                    sampler=sampler if split == 'train' else None, shuffle=False,
                    generator=torch.Generator().manual_seed(fold_seed), num_workers=0,
                    pin_memory=device.type == 'cuda', drop_last=False)
                   for split, dataset in datasets.items()}
        torch.manual_seed(fold_seed)
        model = MirrorLSTM(manifest['input_dim'], weights).to(device)
        fit(model, loaders['train'], loaders['val'], device, directory, epochs,
            learning_rate=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
        model.load_state_dict(torch.load(directory / checkpoint, map_location=device, weights_only=True))
        # Fixed checkpoint: no weighted sampler in train/val/test evaluation.
        train_eval = DataLoader(datasets['train'], batch_size=SETTINGS['batch_size'], shuffle=False, num_workers=0)
        for split, loader in (('train', train_eval), ('val', loaders['val']), ('test', loaders['test'])):
            metrics, cm, predictions = run_epoch(model, loader, device, collect=split == 'test')
            if split == 'test':
                parts = datasets['test'].datasets
                predictions.insert(0, 'test_uid', uid)
                predictions['file'] = np.concatenate([np.repeat(p.path.name, len(p)) for p in parts])
                predictions['target_frame'] = np.concatenate([p.target_frames for p in parts])
                predictions.to_csv(directory / 'test_predictions.csv', index=False)
                save_confusion(cm, directory, f'Initial mirror: UID {uid:02d}, 8 classes')
                best = json.loads((directory / checkpoint.replace('.pth', '.json')).read_text())
                metrics.update(test_uid=uid, best_epoch=best['epoch'])
            else:
                report_dir = directory / f'{split}_evaluation'
                report_dir.mkdir(exist_ok=True)
                save_confusion(cm, report_dir, f'Initial mirror: UID {uid:02d}, {split}, no sampler')
            # test_results is the final completion marker, after all exports.
            write_json(directory / f'{split}_results.json', metrics)
        del model, loaders, train_eval, loader, datasets, parts, predictions, sampler
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        summarize(run_root, ['mirror'])
    print(f'Results saved: {run_root}', flush=True)
    return run_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 16), default=list(range(1, 16)))
    parser.add_argument('--epochs', type=int, default=SETTINGS['epochs'])
    parser.add_argument('--seed', type=int, default=SETTINGS['sampler_seed'])
    parser.add_argument('--selection', choices=['macro_f1', 'val_loss'], default='macro_f1')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(__file__).parent / 'runs' / datetime.now().strftime('MIRROR_LOSO_%Y-%m-%d_%H-%M-%S_%f')
    run_loso(args.data_root, output, args.folds, args.epochs, args.seed, args.selection, args.device, args.cache_dir)


if __name__ == '__main__':
    main()
