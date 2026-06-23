"""
train_contact.py — classificatore binario di contatto strumento-tessuto
========================================================================
Addestra una ResNet18 (pre-addestrata ImageNet) sui crop "punta strumento" etichettati
a mano come contact / no_contact. Gira sul SERVER (torch + GPU). I crop sono quelli
generati da generate_contact_crops.py; le label vengono da annotate_contact.py.

Punti chiave:
  - SPLIT PER VIDEO: i crop dello stesso video non finiscono sia in train sia in val
    (niente leakage: frame dello stesso intervento sono correlati).
  - LOSS PESATA: lo sbilanciamento ~3:1 (contact:no_contact) è compensato con pesi di
    classe inversi alla frequenza, così la rete non "predice sempre contatto".
  - METRICHE PER CLASSE: precision/recall + matrice di confusione + balanced accuracy.
    Il modello migliore è scelto sulla balanced accuracy (non sull'accuratezza grezza,
    che lo sbilanciamento renderebbe ingannevole).

Uso:
  python train_contact.py --data contact_data --epochs 30 --batch 32
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

LABEL2IDX = {"no_contact": 0, "contact": 1}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ContactCrops(Dataset):
    def __init__(self, data_dir, samples, train: bool):
        self.data_dir = data_dir
        self.samples = samples   # lista di (crop_relpath, label_idx)
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, y = self.samples[i]
        img = Image.open(os.path.join(self.data_dir, rel)).convert("RGB")
        return self.tf(img), y


# ─────────────────────────────────────────────────────────────────────────────
# Split per video
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(data_dir):
    manifest = {r["id"]: r for r in json.load(open(os.path.join(data_dir, "manifest.json")))}
    labels   = json.load(open(os.path.join(data_dir, "labels.json")))
    samples = []   # (crop_relpath, label_idx, video)
    for rid, lab in labels.items():
        rec = manifest.get(rid)
        if rec is None or lab not in LABEL2IDX:
            continue
        samples.append((rec["crop"], LABEL2IDX[lab], rec["video"]))
    return samples


def split_by_video(samples, val_frac, seed=0):
    videos = sorted({s[2] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_frac)))
    val_videos = set(videos[:n_val])
    train = [(c, y) for c, y, v in samples if v not in val_videos]
    val   = [(c, y) for c, y, v in samples if v in val_videos]
    return train, val, len(videos), n_val


# ─────────────────────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)   # cm[true, pred]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    rep = {}
    for c in (0, 1):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec  = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        rep[IDX2LABEL[c]] = (prec, rec)
    bal_acc = 0.5 * (rep["no_contact"][1] + rep["contact"][1])  # media dei recall
    return cm, rep, bal_acc


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="contact_data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="worker DataLoader; 0 evita errori di shared memory nei container")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout prima della testa (via forward-hook: non cambia lo state_dict)")
    ap.add_argument("--patience", type=int, default=8,
                    help="early stopping: ferma se la balanced accuracy non migliora da N epoche")
    ap.add_argument("--out", default="checkpoints/contact_resnet18.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = build_samples(args.data)
    train_s, val_s, n_vid, n_val = split_by_video(samples, args.val_frac, args.seed)
    n_pos = sum(1 for _, y in train_s if y == 1)
    n_neg = sum(1 for _, y in train_s if y == 0)
    print(f"Campioni: {len(samples)}  | video: {n_vid} (val {n_val})")
    print(f"Train: {len(train_s)} (contact {n_pos} / no_contact {n_neg})  Val: {len(val_s)}")

    train_dl = DataLoader(ContactCrops(args.data, train_s, True),
                          batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(ContactCrops(args.data, val_s, False),
                          batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # ResNet18 pre-addestrata (fallback se manca internet)
    try:
        from torchvision.models import ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("ResNet18: pesi ImageNet")
    except Exception as e:
        print(f"ResNet18: pesi random (no pretrained: {e})")
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Dropout prima della testa via forward-hook: regolarizza SENZA cambiare le chiavi dello
    # state_dict (resta fc.weight/fc.bias) → i loader di inferenza caricano senza modifiche.
    if args.dropout > 0:
        def _drop_hook(module, inp):
            return (F.dropout(inp[0], p=args.dropout, training=module.training),)
        model.fc.register_forward_pre_hook(_drop_hook)
        print(f"Dropout testa: {args.dropout} (forward-hook)")
    model.to(device)

    # Loss pesata (inverso frequenza) per lo sbilanciamento
    tot = n_pos + n_neg
    w = torch.tensor([tot / (2 * max(n_neg, 1)), tot / (2 * max(n_pos, 1))], dtype=torch.float32, device=device)
    print(f"Pesi classe [no_contact, contact]: {w.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_bal = -1.0
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            tot_loss += loss.item() * x.size(0)
        tot_loss /= len(train_dl.dataset)

        cm, rep, bal = evaluate(model, val_dl, device)
        pc, rc = rep["contact"]
        pn, rn = rep["no_contact"]
        flag = ""
        if bal > best_bal:
            best_bal = bal
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "bal_acc": bal,
                        "label2idx": LABEL2IDX, "mean": _MEAN, "std": _STD}, args.out)
            flag = "  ← best (salvato)"
        else:
            no_improve += 1
        lr_now = optim.param_groups[0]["lr"]
        print(f"[{ep:02d}/{args.epochs}] lr={lr_now:.2e} loss={tot_loss:.4f}  bal_acc={bal:.3f}  "
              f"contact P={pc:.2f} R={rc:.2f}  no_contact P={pn:.2f} R={rn:.2f}{flag}")
        scheduler.step()
        if no_improve >= args.patience:
            print(f"\nEarly stopping: balanced accuracy ferma da {args.patience} epoche (best {best_bal:.3f}).")
            break

    print(f"\nMatrice confusione finale (righe=vero, colonne=pred) [no_contact, contact]:\n{cm}")
    print(f"Miglior balanced accuracy: {best_bal:.3f}  → {args.out}")


if __name__ == "__main__":
    main()
