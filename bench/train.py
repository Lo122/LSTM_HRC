"""Train + evaluate one model on a single subject-disjoint split.

Reports macro-F1 and per-class recall, never accuracy alone -- Screw is ~48% of
frames, so accuracy rewards a model that always says Screw.

NOTE: a single split is only for quick iteration. Measured split-to-split spread
is 0.085 macro-F1 (bench/split_var.py), which is 3.3x the gap between models, so
any comparison between arms must use bench/loso.py instead.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import WindowSet, subject_split, TASK_NAMES, N_TASKS
from models import build
from engine import train_one, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru")
    ap.add_argument("--targets", default="vector", choices=["hard", "vector", "peak"])
    ap.add_argument("--aug", default="none", choices=["none", "mirror"])
    ap.add_argument("--data", default="original")
    ap.add_argument("--aug-dir", default="augmented_mirror")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-uids", default="13,14,15")
    ap.add_argument("--val-uids", default="11,12")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pos-weight-cap", type=float, default=2.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--keep-lift", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te_u = {int(x) for x in a.test_uids.split(",")}
    va_u = {int(x) for x in a.val_uids.split(",")}
    paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    aug = sorted(glob.glob(os.path.join(a.aug_dir, "*.pt"))) if a.aug == "mirror" else None
    tr_p, va_p, te_p = subject_split(paths, te_u, va_u, aug_paths=aug)
    n_aug = sum(1 for p in tr_p if "augmented" in p)
    print("[split] train %d files (%d augmented) / val %d / test %d"
          "  test uids %s, val uids %s"
          % (len(tr_p), n_aug, len(va_p), len(te_p), sorted(te_u), sorted(va_u)))

    t0 = time.time()
    tr = WindowSet(tr_p, a.win, a.hop, drop_lift=not a.keep_lift)
    va = WindowSet(va_p, a.win, a.hop, stats=tr.stats, drop_lift=not a.keep_lift)
    te = WindowSet(te_p, a.win, a.hop, stats=tr.stats, drop_lift=not a.keep_lift)
    Xtr, ytr = tr.tensors(); Xva, yva = va.tensors(); Xte, yte = te.tensors()
    print("[data] windows train %d val %d test %d  dim %d  (%.0fs)"
          % (len(Xtr), len(Xva), len(Xte), Xtr.shape[-1], time.time() - t0))
    print("[data] background frames in test: %.1f%%"
          % (100 * yte["bg"].numpy().mean()))

    model = build(a.model, Xtr.shape[-1], n_tasks=ytr["plateau"].shape[1]).to(dev)
    model, best_val, state = train_one(model, Xtr, ytr, Xva, yva, dev, a.targets,
                                       a.epochs, a.bs, a.lr,
                                       pos_weight_cap=(a.pos_weight_cap or None),
                                       focal_gamma=a.focal_gamma)
    res = evaluate(model, Xte, yte, dev, a.targets)
    res.update(model=a.model, targets=a.targets, aug=a.aug, val_macro_f1=best_val,
               test_uids=sorted(te_u), n_params=sum(p.numel() for p in model.parameters()))

    print("\n=== TEST (held-out subjects, background-masked) ===")
    print("  macro-F1      %.3f" % res["macro_f1"])
    print("  balanced-acc  %.3f   (chance %.3f)" % (res["balanced_acc"], 1 / N_TASKS))
    print("  accuracy      %.3f   <- do not report this alone" % res["accuracy"])
    if "lane_ap" in res:
        print("  lane AP       %.3f   (multi-label view)" % res["lane_ap"])
        print("  lane F1       %.3f" % res["lane_f1"])
        print("  background F1 %.3f" % res["bg_f1"])
    print("  mistake F1    %.3f" % res["mistake_f1"])
    print("  progress MAE  %.1f / 100" % res["progress_mae"])
    print("  per-class recall:")
    for k, v in res["per_class_recall"].items():
        print("    %-12s %.2f" % (k, v))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        torch.save({"state_dict": state, "stats": tr.stats, "config": vars(a),
                    "dim": Xtr.shape[-1]}, a.out.replace(".json", ".pth"))
        print("\n[saved] %s + .pth" % a.out)


if __name__ == "__main__":
    main()
