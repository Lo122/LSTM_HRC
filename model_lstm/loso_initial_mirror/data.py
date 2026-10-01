"""Cache raw features once; fit original mean/std separately for each fold."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from data_proc_2d.app.build_norm_dataset_tune import l1_file_splits, l1_subject_split, normalization_id

REFERENCE = json.loads(Path(__file__).with_name('reference_config.json').read_text())
SETTINGS = {k: REFERENCE[k] for k in (
    'epochs', 'batch_size', 'window_size', 'stride', 'predict_offset', 'mode',
    'sampler_seed', 'dropout', 'num_layers', 'weighted_loss', 'weight_decay',
    'lambda_step', 'lambda_progress', 'lambda_mistake', 'input_dim',
)}
SETTINGS.update(lr=REFERENCE['learning_rate'], hidden_layer=REFERENCE['hidden_dim'],
                x_keys=REFERENCE['feature_keys'], class7_loss_factor=1.0,
                step_weighting='sqrt_inverse_frequency',
                reference_experiment='exp_2026-09-24_17-02-55',
                sampler_interpretation='sqrt inverse frequency per test_notes; saved sampler_weighting conflicts')
FEATURE_KEYS = SETTINGS['x_keys']


def source_identity(splits):
    return {split: [dict(path=str(p.resolve()), size=p.stat().st_size,
                        mtime_ns=p.stat().st_mtime_ns) for p in paths] for split, paths in splits.items()}


class RawRecording:
    def __init__(self, path, cache_dir):
        self.path = Path(path)
        stat = self.path.stat()
        identity = [str(self.path.resolve()), stat.st_size, stat.st_mtime_ns, FEATURE_KEYS, 'initial_mirror_v1']
        directory = Path(cache_dir) / hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        marker = directory / 'ready.json'
        if not marker.exists():
            data = torch.load(self.path, map_location='cpu', weights_only=False)
            features = [data['features'][key].numpy().astype(np.float32, copy=False) for key in FEATURE_KEYS]
            if any(x.ndim != 2 or len(x) != len(features[0]) or not np.isfinite(x).all() for x in features):
                raise ValueError(f'Invalid/nonfinite raw features: {path}; no silent frame removal')
            step = data['labels']['task_id'].numpy().astype(np.int64)
            step = np.where(step == 8, 7, step)  # Same background remapping, without modifying PT.
            progress = data['labels']['task_progress'].numpy().astype(np.float32) / 100.
            mistake = data['labels']['mistake'].numpy().astype(np.int64)
            if (any(y.shape != (len(features[0]),) for y in (step, progress, mistake))
                    or not np.isin(step, range(8)).all() or not np.isin(mistake, [0, 1]).all()
                    or not np.isfinite(progress).all()):
                raise ValueError(f'Invalid labels: {path}')
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in dict(features=np.concatenate(features, axis=1), step=step,
                                    progress=progress, mistake=mistake).items():
                temporary = directory / f'{name}.tmp'
                with temporary.open('wb') as stream:
                    np.save(stream, value, allow_pickle=False)
                temporary.replace(directory / f'{name}.npy')
            temporary = directory / 'ready.tmp'
            temporary.write_text(json.dumps(dict(dimensions=[x.shape[1] for x in features])))
            temporary.replace(marker)
        self.dimensions = json.loads(marker.read_text())['dimensions']
        self.features, self.step, self.progress, self.mistake = [
            np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            for name in ('features', 'step', 'progress', 'mistake')]


def fit_norm(recordings):
    """Same float32 all-frame population mean/std+1e-6 as the notebook exporter.

    Concatenate one feature group at a time to bound peak memory.
    """
    if not recordings:
        raise ValueError('Empty training recordings')
    dimensions = recordings[0].dimensions
    if any(r.dimensions != dimensions for r in recordings):
        raise ValueError('Feature layout mismatch')
    stats, start = {}, 0
    for key, width in zip(FEATURE_KEYS, dimensions):
        values = np.concatenate([r.features[:, start:start + width] for r in recordings])
        stats[f'{key}_mean'] = values.mean(axis=0).astype(np.float32)
        stats[f'{key}_std'] = (values.std(axis=0) + 1e-6).astype(np.float32)
        start += width
        del values
    return stats


class WindowDataset(Dataset):
    def __init__(self, recording, stats):
        self.recording, self.path = recording, recording.path
        self.mean = np.concatenate([stats[f'{k}_mean'] for k in FEATURE_KEYS])
        std = np.concatenate([stats[f'{k}_std'] for k in FEATURE_KEYS])
        self.std = np.where(std < 1e-6, 1., std)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid fold normalization')
        self.starts = np.arange(0, max(0, len(recording.step) - SETTINGS['window_size'] + 1), SETTINGS['stride'])
        self.target_frames = self.starts + SETTINGS['window_size'] - 1

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start, target = self.starts[index], self.target_frames[index]
        r = self.recording
        x = ((r.features[start:target + 1] - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(x), dict(step=torch.tensor(r.step[target], dtype=torch.long),
            progress=torch.tensor(r.progress[target], dtype=torch.float32),
            mistake=torch.tensor(r.mistake[target], dtype=torch.long))


def load_fold(data_root, fold_dir, uid, cache_dir):
    splits = l1_file_splits(data_root, uid)
    identity = dict(profile='initial_mirror_raw_v1', feature_keys=FEATURE_KEYS,
                    subjects=l1_subject_split(uid), sources=source_identity(splits),
                    statistics_frame_step=1, statistics_dtype='float32', std_ddof=0)
    fold_dir = Path(fold_dir)
    manifest_path, norm_path = fold_dir / 'dataset_manifest.json', fold_dir / 'norm_stats.npz'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError('Source data or normalization protocol changed')
        if hashlib.sha256(norm_path.read_bytes()).hexdigest() != manifest['normalization_sha256']:
            raise ValueError('Normalization hash mismatch')
    recordings = {split: [RawRecording(p, cache_dir) for p in paths] for split, paths in splits.items()}
    dimensions = recordings['train'][0].dimensions
    if sum(dimensions) != SETTINGS['input_dim']:
        raise ValueError('Input dimension differs from reference config')
    if any(r.dimensions != dimensions for items in recordings.values() for r in items):
        raise ValueError('Feature layout mismatch across splits')
    if manifest:
        with np.load(norm_path) as archive:
            stats = dict(archive)
    else:
        stats = fit_norm(recordings['train'])
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / 'norm_stats.tmp').open('wb') as stream:
            np.savez(stream, **stats)
        (fold_dir / 'norm_stats.tmp').replace(norm_path)
        manifest = dict(identity, input_dim=sum(dimensions), dimensions=dimensions,
                        normalization_id=normalization_id(stats),
                        normalization_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest())
        temporary = fold_dir / 'dataset_manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
    datasets = {split: ConcatDataset([WindowDataset(r, stats) for r in items]) for split, items in recordings.items()}
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError('Empty split')
    return datasets, manifest


def training_weights(dataset, seed):
    targets = np.concatenate([d.recording.step[d.target_frames] for d in dataset.datasets])
    counts = np.bincount(targets, minlength=8)
    if (counts == 0).any():
        raise ValueError(f'Missing training classes: {np.flatnonzero(counts == 0).tolist()}')
    weights = 1. / np.sqrt(counts[targets])
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(targets), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    class_weights = 1. / np.sqrt(counts)
    class_weights /= np.average(class_weights, weights=counts)
    class_weights[7] *= SETTINGS['class7_loss_factor']
    return sampler, counts.tolist(), class_weights.tolist()
