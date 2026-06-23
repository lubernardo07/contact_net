"""
eval_contact.py — valuta un checkpoint contact_net sullo STESSO val split
==========================================================================
Confronto apples-to-apples fra checkpoint diversi (es. contact_resnet18.pt vs _v2.pt):
ricostruisce lo stesso split per-video (stesso --seed/--val_frac di train_contact) e
stampa balanced accuracy + matrice di confusione + P/R per classe. Niente training.

Uso (SERVER):
  python eval_contact.py --data contact_data --net checkpoints/contact_resnet18.pt
  python eval_contact.py --data contact_data --net checkpoints/contact_resnet18_v2.pt

CAVEAT: il val split è ricavato dai label ATTUALI (dataset merge). Per il modello VECCHIO
alcuni video di questo val potrebbero essere stati nel SUO training → vantaggio a favore del
vecchio. Quindi: se il nuovo pareggia o vince, è solidamente migliore.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from train_contact import build_samples, split_by_video, evaluate, ContactCrops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="contact_data")
    ap.add_argument("--net", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.net, map_location=device, weights_only=False)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    samples = build_samples(args.data)
    _train_s, val_s, n_vid, n_val = split_by_video(samples, args.val_frac, args.seed)
    val_dl = DataLoader(ContactCrops(args.data, val_s, False),
                        batch_size=args.batch, shuffle=False)

    cm, rep, bal = evaluate(model, val_dl, device)
    pc, rc = rep["contact"]
    pn, rn = rep["no_contact"]
    print(f"Checkpoint: {args.net}  (bal_acc salvato nel file = {ck.get('bal_acc', '?')})")
    print(f"Val: {len(val_s)} crop su {n_val}/{n_vid} video (seed {args.seed}, val_frac {args.val_frac})")
    print(f"balanced accuracy : {bal:.3f}")
    print(f"  contact     P={pc:.3f} R={rc:.3f}")
    print(f"  no_contact  P={pn:.3f} R={rn:.3f}")
    print(f"matrice confusione (righe=vero, colonne=pred) [no_contact, contact]:\n{cm}")


if __name__ == "__main__":
    main()
