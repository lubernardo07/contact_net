"""
select_hard_crops.py — selezione MIRATA dei crop da annotare (active learning)
================================================================================
Usa il contact_net ATTUALE per predire su tutti i crop del manifest e tiene solo i
NON ancora annotati su cui la rete è più INCERTA (prob di contatto ~0.5): sono i casi
"vicino ma non tocca" dove nuove label migliorano di più il gate. Niente rigenerazione:
sfrutta i crop già presenti sul server.

Produce una cartella piccola (--out) con SOLO i crop selezionati (manifest + crops/zoom/
context), pronta da comprimere e annotare in locale col GUI. Le label esistenti restano
nella cartella grande; dopo aver annotato i nuovi, si fondono per il training.

Uso (SERVER, GPU):
  python select_hard_crops.py --data contact_data --net checkpoints/contact_resnet18.pt \
      --k 500 --out contact_data_hard
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="contact_data")
    ap.add_argument("--net",  default="checkpoints/contact_resnet18.pt")
    ap.add_argument("--k",    type=int, default=500, help="quanti crop incerti tenere")
    ap.add_argument("--band", type=float, default=0.0,
                    help="se >0, considera solo prob in [0.5-band, 0.5+band] (oltre al top-k)")
    ap.add_argument("--out",  default="contact_data_hard")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ck = torch.load(args.net, map_location=device, weights_only=False)  # checkpoint nostro (contiene scalari numpy)
    mean = ck.get("mean", [0.485, 0.456, 0.406])
    std  = ck.get("std",  [0.229, 0.224, 0.225])
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    lpath = os.path.join(args.data, "labels.json")
    spath = os.path.join(args.data, "skips.json")
    labels = json.load(open(lpath)) if os.path.isfile(lpath) else {}
    skips  = set(json.load(open(spath))) if os.path.isfile(spath) else set()
    done = set(labels) | skips

    todo = [r for r in manifest if r["id"] not in done]
    print(f"manifest {len(manifest)}, già fatti {len(done)}, da valutare {len(todo)}")

    # Inferenza a batch sui crop non ancora annotati
    probs = {}
    buf_img, buf_id = [], []

    def flush():
        if not buf_img:
            return
        x = torch.stack(buf_img).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        for i, rid in enumerate(buf_id):
            probs[rid] = float(p[i])
        buf_img.clear()
        buf_id.clear()

    for n, r in enumerate(todo):
        cp = os.path.join(args.data, r["crop"])
        if not os.path.isfile(cp):
            continue
        buf_img.append(tf(Image.open(cp).convert("RGB")))
        buf_id.append(r["id"])
        if len(buf_img) >= args.batch:
            flush()
        if (n + 1) % 2000 == 0:
            print(f"  valutati {n+1}/{len(todo)}…")
    flush()

    # Ordina per incertezza: |prob - 0.5| crescente (più incerto prima)
    ranked = sorted(probs.items(), key=lambda kv: abs(kv[1] - 0.5))
    if args.band > 0:
        ranked = [(rid, p) for rid, p in ranked if abs(p - 0.5) <= args.band]
    sel_ids = {rid for rid, _ in ranked[:args.k]}
    if sel_ids:
        ps = [probs[i] for i in sel_ids]
        print(f"selezionati {len(sel_ids)} crop incerti  "
              f"(prob contatto: min {min(ps):.2f} / max {max(ps):.2f} / media {np.mean(ps):.2f})")

    # Cartella mirata: solo i record selezionati + le loro immagini
    os.makedirs(args.out, exist_ok=True)
    for sub in ("crops", "zoom", "context"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    new_manifest = []
    for r in manifest:
        if r["id"] not in sel_ids:
            continue
        new_manifest.append(r)
        for key in ("crop", "zoom", "context"):
            rel = r.get(key)
            if rel and os.path.isfile(os.path.join(args.data, rel)):
                shutil.copy(os.path.join(args.data, rel), os.path.join(args.out, rel))
    json.dump(new_manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    json.dump({}, open(os.path.join(args.out, "labels.json"), "w"))   # tutti da annotare
    print(f"\n── cartella mirata pronta: {args.out}/  ({len(new_manifest)} crop da annotare)")
    print(f"   comprimi e scarica:  tar -czf {args.out}.tar.gz {args.out}")


if __name__ == "__main__":
    main()
