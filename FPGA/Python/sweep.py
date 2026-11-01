

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import FootDropDataset
from model import FootDropCNN

REPO_ROOT   = Path(__file__).resolve().parent
CACHE_ROOT  = REPO_ROOT / "sweep_cache"
RESULTS     = REPO_ROOT / "sweep_results.json"
EXTRACTOR   = REPO_ROOT / "extract_dataset_sliding.py"

SWEEP_CUTOFFS = [20, 30, 40]
WIDTHS = [
    ("base_16_32", 16, 32, 32),
    ("wide_24_48", 24, 48, 32),
]
DEFAULT_PANEL = ("AB185", "AB188", "AB156")

EPOCHS       = 40
LR           = 1e-3
BATCH_SIZE   = 64
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
SEED         = 123
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dataset(cutoff_hz):
    out = CACHE_ROOT / f"env{cutoff_hz}hz"
    if (out / "inputs.npy").exists():
        print(f"  [cache hit] {out.relative_to(REPO_ROOT)}")
        return out
    print(f"  [extracting] env-cutoff={cutoff_hz} Hz -> {out.relative_to(REPO_ROOT)} ...")
    subprocess.run(
        [sys.executable, str(EXTRACTOR), "--env-cutoff", str(cutoff_hz), "--out", str(out)],
        check=True,
    )
    return out


def train_eval(data_dir, val_subject, subjects, c1, c2, hidden, epochs, tag=""):
    train_ds = FootDropDataset(str(data_dir), (val_subject,))
    val_ds   = FootDropDataset(str(data_dir), tuple(s for s in subjects if s != val_subject))

    torch.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_pos = (train_ds.labels == 1).sum().item()
    num_neg = (train_ds.labels == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = FootDropCNN(c1=c1, c2=c2, lstm_hidden=hidden, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best = (0.0, 0.0, 0.0)
    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(inputs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            n_batches += 1
        avg_loss = running_loss / max(n_batches, 1)

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for vin, vlab in val_loader:
                logits = model(vin.to(DEVICE)).squeeze(1)
                preds.append((logits > 0).int().cpu())
                gts.append(vlab.int())
        preds = torch.cat(preds).numpy()
        gts   = torch.cat(gts).numpy()
        f1 = f1_score(gts, preds, zero_division=0)
        if f1 > best[0]:
            best = (f1,
                    precision_score(gts, preds, zero_division=0),
                    recall_score(gts, preds, zero_division=0))

        print(f"    {tag} epoch {epoch+1:>2}/{epochs}  loss={avg_loss:.4f}  "
              f"val_F1={f1:.3f}  (best {best[0]:.3f})", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description="Joint envelope-cutoff x width sweep.")
    ap.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL),
                    help="Held-out subjects to average over (space-separated).")
    ap.add_argument("--cutoffs", nargs="+", type=float, default=SWEEP_CUTOFFS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--quick", action="store_true",
                    help="Only the baseline width (skip the wide variant).")
    args = ap.parse_args()

    widths = WIDTHS[:1] if args.quick else WIDTHS
    print(f"Device: {DEVICE} | cutoffs={args.cutoffs} | widths={[w[0] for w in widths]} "
          f"| panel={args.panel} | epochs={args.epochs}\n")

    results = []
    for cutoff in args.cutoffs:
        data_dir = ensure_dataset(cutoff)
        subjects = sorted({str(s) for s in np.load(data_dir / "subject_ids.npy", allow_pickle=True)})
        for wname, c1, c2, hidden in widths:
            for val_subject in args.panel:
                tag = f"[{cutoff}Hz {wname} ✗{val_subject}]"
                print(f"\n  >>> training {tag}  ({args.epochs} epochs)", flush=True)
                f1, prec, rec = train_eval(data_dir, val_subject, subjects,
                                           c1, c2, hidden, args.epochs, tag=tag)
                print(f"  cutoff={cutoff:>4} Hz | {wname:11s} | hold-out {val_subject} "
                      f"-> F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}", flush=True)
                results.append(dict(cutoff=cutoff, width=wname, val_subject=val_subject,
                                    f1=f1, precision=prec, recall=rec))

    print("\n" + "=" * 60)
    print("MEAN PANEL F1 per (cutoff, width)")
    print("=" * 60)
    agg = {}
    for r in results:
        agg.setdefault((r["cutoff"], r["width"]), []).append(r["f1"])
    table = [(c, w, float(np.mean(v))) for (c, w), v in agg.items()]
    table.sort(key=lambda t: t[2], reverse=True)
    for c, w, mean_f1 in table:
        print(f"  cutoff={c:>4} Hz | {w:11s} | mean F1 = {mean_f1:.3f}")

    best_c, best_w, best_f1 = table[0]
    print(f"\nBEST: cutoff={best_c} Hz, width={best_w}  (mean panel F1 {best_f1:.3f})")
    print("Next: confirm this config with a FULL 10-fold LOSO before trusting it "
          "(the panel is a selector, not the final estimate).")

    RESULTS.write_text(json.dumps(
        {"config": {"cutoffs": args.cutoffs, "widths": [w[0] for w in widths],
                    "panel": args.panel, "epochs": args.epochs, "device": DEVICE},
         "rows": results,
         "mean_panel_f1": [{"cutoff": c, "width": w, "mean_f1": f} for c, w, f in table],
         "best": {"cutoff": best_c, "width": best_w, "mean_f1": best_f1}},
        indent=2))
    print(f"\nSaved {RESULTS.name}")


if __name__ == "__main__":
    main()
