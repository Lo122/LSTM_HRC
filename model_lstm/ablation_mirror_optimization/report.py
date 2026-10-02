"""Validate the fixed control and compare only matched held-out participants."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def validate_baseline(baseline_run, protocol, folds):
    root = Path(baseline_run)
    baseline = json.loads((root / 'protocol.json').read_text())
    differences = [k for k, value in protocol.items()
                   if k not in ('folds', 'device') and baseline.get(k) != value]
    if differences:
        raise ValueError(f'Baseline settings differ: {differences}')
    for uid in folds:
        directory = root / 'mirror' / f'fold_{uid:02d}'
        for name in ('config.json', 'norm_stats.npz', 'dataset_manifest.json',
                     'test_results.json', 'test_predictions.csv'):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        config = json.loads((directory / 'config.json').read_text())
        if any(config.get(k) != value for k, value in protocol.items() if k not in ('folds', 'device')):
            raise ValueError(f'Baseline fold configuration differs: {directory}')
        manifest = json.loads((directory / 'dataset_manifest.json').read_text())
        if hashlib.sha256((directory / 'norm_stats.npz').read_bytes()).hexdigest() != manifest['normalization_sha256']:
            raise ValueError(f'Baseline normalization hash mismatch: {directory}')


def compare_baseline(run_root, baseline_run):
    root, baseline = Path(run_root), Path(baseline_run)
    metrics = ('macro_f1_8', 'accuracy_8', 'background_f1', 'task_macro_f1_7_conditional')
    rows = []
    for path in sorted((root / 'mirror').glob('fold_*/test_results.json')):
        control = baseline / 'mirror' / path.parent.name
        current = json.loads(path.read_text())
        previous = json.loads((control / 'test_results.json').read_text())
        for name in ('norm_stats.npz', 'dataset_manifest.json'):
            if (path.parent / name).read_bytes() != (control / name).read_bytes():
                raise ValueError(f'Normalization/source metadata differs from control: {path.parent}')
        identity = ['test_uid', 'file', 'target_frame', 'true_step']
        a = pd.read_csv(control / 'test_predictions.csv', usecols=identity)
        b = pd.read_csv(path.parent / 'test_predictions.csv', usecols=identity)
        if not a.equals(b):
            raise ValueError(f'Test windows or truth differ from control: {path.parent}')
        row = dict(test_uid=current['test_uid'])
        for key in metrics:
            row[f'baseline_{key}'] = previous[key]
            row[f'l1opt_{key}'] = current[key]
            row[f'delta_{key}'] = current[key] - previous[key]
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values('test_uid')
    frame.to_csv(root / 'paired_optimizer_comparison.csv', index=False)
    summary = dict(baseline_run=str(baseline), completed_pairs=len(frame),
                   test_uids=frame.test_uid.tolist(), std_ddof=0,
                   delta_direction='L1 optimization minus initial mirror', metrics={})
    for key in metrics:
        summary['metrics'][key] = {
            label: dict(mean=float(frame[f'{label}_{key}'].mean()),
                        std=float(frame[f'{label}_{key}'].std(ddof=0)))
            for label in ('baseline', 'l1opt', 'delta')}
    (root / 'optimizer_comparison_summary.json').write_text(json.dumps(summary, indent=2))
