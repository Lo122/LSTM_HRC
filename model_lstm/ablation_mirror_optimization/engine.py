"""L1 optimization settings applied to the unchanged initial-mirror objective."""
import csv
import json
from pathlib import Path
import torch
from model_lstm.ablation_step_supervision.engine import run_epoch

OPTIMIZATION = dict(optimizer='AdamW', lr=1e-3, weight_decay=1e-4,
                    scheduler='CosineAnnealingLR', gradient_clip=1.0)


def fit(model, train_loader, val_loader, device, run_dir, epochs):
    run_dir = Path(run_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZATION['lr'],
                                 weight_decay=OPTIMIZATION['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
    best_f1, best_loss = -1., float('inf')
    for epoch in range(1, epochs + 1):
        train, _, _ = run_epoch(model, train_loader, device, optimizer,
                                gradient_clip=OPTIMIZATION['gradient_clip'])
        val, _, _ = run_epoch(model, val_loader, device)
        row = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'],
                   **{f'train_{k}': v for k, v in train.items()},
                   **{f'val_{k}': v for k, v in val.items()})
        with (run_dir / 'metrics.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        if val['macro_f1_8'] > best_f1:
            best_f1 = val['macro_f1_8']
            torch.save(model.state_dict(), run_dir / 'best_macro_f1.pth')
            (run_dir / 'best_macro_f1.json').write_text(json.dumps(dict(epoch=epoch, **val), indent=2))
        if val['loss'] < best_loss:
            best_loss = val['loss']
            torch.save(model.state_dict(), run_dir / 'best_val_loss.pth')
            (run_dir / 'best_val_loss.json').write_text(json.dumps(dict(epoch=epoch, **val), indent=2))
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}: lr {row['learning_rate']:.6g}, "
              f"val F1 {val['macro_f1_8']:.4f}, val loss {val['loss']:.4f}", flush=True)
    model.load_state_dict(torch.load(run_dir / 'best_macro_f1.pth', map_location=device, weights_only=True))
