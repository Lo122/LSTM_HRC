"""Pooled confusion matrices across all 15 LOSO folds.

Every fold contributes its own held-out subject, so the matrix below covers all
15 people -- not one lucky split. Predictions are background-masked and the
ground-truth argmax is peak-tie-broken, matching engine.evaluate().
"""
import sys, os, glob, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import WindowSet, take_key, task_names, ALL_TASK_NAMES
from models import build
from engine import predict
from sklearn.metrics import confusion_matrix

dev = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob("original/*.pt"))


def pooled(tag):
    drop = "7class" not in tag
    names = task_names(drop)
    C = len(names)
    cm = np.zeros((C, C), dtype=np.int64)
    for fp in sorted(glob.glob("bench/results/folds/%s_uid*.pth" % tag)):
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        te_p = [p for p in paths if take_key(p)[0] in set(ck["test_uids"])]
        te = WindowSet(te_p, ck["win"], ck["hop"], stats=ck["stats"], drop_lift=drop)
        X, y = te.tensors()
        m = build(ck["model"], ck["dim"], n_tasks=C).to(dev)
        m.load_state_dict(ck["state_dict"])
        logit = predict(m, X, dev)["task"].numpy()
        pl = y["plateau"].numpy(); pk = y["peak"].numpy()
        keep = ~y["bg"].numpy().astype(bool)
        true = (pl + 1e-3 * pk).argmax(1)[keep]
        pred = logit.argmax(1)[keep]
        cm += confusion_matrix(true, pred, labels=range(C))
    return names, cm


def show(tag, names, cm):
    C = len(names)
    row = cm.sum(1, keepdims=True)
    col = cm.sum(0)
    norm = cm / np.maximum(row, 1)
    print("\n" + "=" * 74)
    print("%s   (%d windows, all 15 held-out subjects pooled)" % (tag, cm.sum()))
    print("=" * 74)
    print("\nROW-NORMALISED  (row = true class, so the diagonal is RECALL)")
    print("%-13s" % "true \ pred" + "".join("%9s" % n[:8] for n in names) + "%9s" % "support")
    for i, n in enumerate(names):
        print("%-13s" % n[:12] + "".join("%9.2f" % v for v in norm[i]) + "%9d" % row[i, 0])
    print("\nPER-CLASS")
    print("%-13s %8s %10s %8s %9s" % ("class", "recall", "precision", "F1", "support"))
    f1s = []
    for i, n in enumerate(names):
        rec = cm[i, i] / max(row[i, 0], 1)
        pre = cm[i, i] / max(col[i], 1)
        f1 = 2 * rec * pre / max(rec + pre, 1e-9)
        f1s.append(f1)
        print("%-13s %8.2f %10.2f %8.2f %9d" % (n[:12], rec, pre, f1, row[i, 0]))
    print("%-13s %8s %10s %8.3f %9d"
          % ("MACRO", "", "", float(np.mean(f1s)), cm.sum()))
    print("accuracy %.3f" % (np.trace(cm) / max(cm.sum(), 1)))
    # biggest off-diagonal confusions
    off = [(norm[i, j], names[i], names[j], cm[i, j])
           for i in range(C) for j in range(C) if i != j]
    off.sort(reverse=True)
    print("\nlargest confusions:")
    for v, a, b, n in off[:4]:
        print("   %-12s -> %-12s %.0f%%  (%d windows)" % (a, b, 100 * v, n))
    return {"names": names, "matrix": cm.tolist(),
            "row_normalised": np.round(norm, 4).tolist()}


out = {}
for tag in ["R0_7class", "R2_cap"]:
    names, cm = pooled(tag)
    out[tag] = show(tag, names, cm)
json.dump(out, open("bench/results/confusion_matrices.json", "w"), indent=2)
print("\n[saved] bench/results/confusion_matrices.json")
