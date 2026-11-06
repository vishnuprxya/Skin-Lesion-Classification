"""Train + evaluate on HAM10000 (Kaggle: kmader/skin-cancer-mnist-ham10000).

python train.py --data_dir ./data/ham10000 --epochs 30 --out_dir ./outputs
(small mode) python train.py --data_dir ./data/ham10000 --pixels_csv hmnist_28_28_RGB.csv --side 28 --img_size 128
"""
import argparse, json, os, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from model import HybridCNNTransformer, CLASSES, MEAN, STD, eval_transform


class LesionDS(Dataset):
    """Reads either JPG files (r.path) or rows of an in-memory pixel array (images[r.row])."""
    def __init__(self, df, tf, images=None):
        self.df, self.tf, self.images = df.reset_index(drop=True), tf, images
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = (Image.fromarray(self.images[int(r.row)]) if self.images is not None
               else Image.open(r.path).convert("RGB"))
        return self.tf(img), int(r.label)


def load_pixels_csv(data_dir, pixels_csv, side):
    """Light-weight HAM10000: hmnist_{side}_{side}_RGB.csv (+ HAM10000_metadata.csv). No JPGs needed."""
    meta = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
    pix = pd.read_csv(os.path.join(data_dir, pixels_csv))
    assert len(meta) == len(pix), "metadata and pixel CSV have different row counts"
    # Safety check: rows must be aligned. Each dx must map to exactly one HMNIST label.
    ct = pd.crosstab(meta["dx"].values, pix["label"].values)
    purity = ct.max(axis=1).sum() / ct.values.sum()
    if purity < 0.99:
        raise ValueError(f"Row order of metadata and pixel CSV does not match (purity={purity:.3f}).")
    images = pix.drop(columns="label").values.astype(np.uint8).reshape(-1, side, side, 3)
    meta["row"] = np.arange(len(meta))
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return meta, images


def load_metadata(data_dir):
    meta = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)}
    meta["path"] = meta["image_id"].map(paths)
    meta = meta.dropna(subset=["path"]).copy()
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return meta


def split(meta, seed=42):
    """Split BY lesion_id so the same lesion never appears in train and test (no leakage)."""
    g = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, tmp = next(g.split(meta, groups=meta.lesion_id))
    tmp_df = meta.iloc[tmp]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(g2.split(tmp_df, groups=tmp_df.lesion_id))
    return meta.iloc[tr], tmp_df.iloc[va], tmp_df.iloc[te]


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev)).softmax(-1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pixels_csv", default=None,
                    help="e.g. hmnist_28_28_RGB.csv -> small-download mode (no JPG folders needed)")
    ap.add_argument("--side", type=int, default=64, help="image side of the pixel CSV (28 or 64)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.pixels_csv:
        meta, images = load_pixels_csv(a.data_dir, a.pixels_csv, a.side)
    else:
        meta, images = load_metadata(a.data_dir), None
    tr, va, te = split(meta)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | device {dev}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(a.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ev_tf = eval_transform(a.img_size)

    # class imbalance: weighted sampler (nv is ~67% of the data)
    counts = tr.label.value_counts().reindex(range(len(CLASSES))).fillna(1).values
    sw = tr.label.map(lambda l: 1.0 / counts[l]).values
    sampler = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(LesionDS(tr, train_tf, images), a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True)
    dl_va = DataLoader(LesionDS(va, ev_tf, images), a.batch * 2, num_workers=a.workers)
    dl_te = DataLoader(LesionDS(te, ev_tf, images), a.batch * 2, num_workers=a.workers)

    model = HybridCNNTransformer(len(CLASSES), a.img_size, a.dim, a.depth, a.heads).to(dev)
    bb = list(model.backbone.parameters()); bb_ids = {id(p) for p in bb}
    rest = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr / 3}, {"params": rest, "lr": a.lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    best, ck_path, hist = -1, os.path.join(a.out_dir, "best_model.pt"), []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0
        for x, y in tqdm(dl_tr, desc=f"epoch {ep}/{a.epochs}", leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        P, Y = predict(model, dl_va, dev)
        f1 = f1_score(Y, P.argmax(1), average="macro"); acc = (P.argmax(1) == Y).mean()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_acc": float(acc), "val_macro_f1": float(f1)})
        print(f"ep {ep}: loss {tot/len(tr):.4f} val_acc {acc:.4f} val_macroF1 {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "config": dict(img_size=a.img_size, dim=a.dim, depth=a.depth, heads=a.heads)}, ck_path)

    # ---- final test evaluation with best checkpoint
    model.load_state_dict(torch.load(ck_path, map_location=dev)["model"])
    P, Y = predict(model, dl_te, dev); pred = P.argmax(1)
    rep = classification_report(Y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cmx = confusion_matrix(Y, pred, labels=range(len(CLASSES)))
    try: auc = float(roc_auc_score(Y, P, multi_class="ovr", average="macro"))
    except ValueError: auc = None
    metrics = {"accuracy": float((pred == Y).mean()), "balanced_accuracy": float(balanced_accuracy_score(Y, pred)),
               "macro_f1": float(f1_score(Y, pred, average="macro")), "macro_auc_ovr": auc,
               "report": rep, "confusion_matrix": cmx.tolist(), "history": hist, "classes": CLASSES}
    json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    plt.figure(figsize=(7, 6)); sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "confusion_matrix.png"), dpi=150)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("report", "confusion_matrix", "history", "classes")}, indent=2))


if __name__ == "__main__":
    main()
