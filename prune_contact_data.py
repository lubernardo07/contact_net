"""
prune_contact_data.py — pota un contact_data al solo set ANNOTATO
==================================================================
Il pool grande sul server (es. 16k+ crop generati con --append) è troppo grande e in
gran parte inutile/corrotto: non verrà mai annotato. Questo script tiene SOLO i crop già
annotati (id in labels.json) e cancella tutti gli altri da crops/zoom/context, riscrivendo
manifest.json di conseguenza. labels.json e skips.json restano intatti.

SICUREZZA: di default è DRY-RUN (non cancella nulla, stampa solo cosa farebbe). Cancella
davvero solo con --apply. Le annotazioni (labels.json) non vengono mai toccate.

Uso (SERVER):
  python prune_contact_data.py --data contact_data            # anteprima
  python prune_contact_data.py --data contact_data --apply     # esegue
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="contact_data")
    ap.add_argument("--keep_skips", action="store_true",
                    help="tieni anche i crop SALTATI (in skips.json), non solo gli annotati")
    ap.add_argument("--apply", action="store_true", help="esegue le cancellazioni (default: dry-run)")
    args = ap.parse_args()

    manifest_path = os.path.join(args.data, "manifest.json")
    labels_path   = os.path.join(args.data, "labels.json")
    skips_path    = os.path.join(args.data, "skips.json")

    manifest = json.load(open(manifest_path))
    labels   = json.load(open(labels_path)) if os.path.isfile(labels_path) else {}
    skips    = set(json.load(open(skips_path))) if os.path.isfile(skips_path) else set()

    keep = set(labels)
    if args.keep_skips:
        keep |= skips

    keep_records = [r for r in manifest if r["id"] in keep]
    drop_records = [r for r in manifest if r["id"] not in keep]

    print(f"manifest totale : {len(manifest)}")
    print(f"annotati (labels): {len(labels)}   saltati (skips): {len(skips)}")
    print(f"DA TENERE        : {len(keep_records)}")
    print(f"DA CANCELLARE    : {len(drop_records)}")

    # File da cancellare (crop/zoom/context dei record droppati)
    to_delete = []
    for r in drop_records:
        for key in ("crop", "zoom", "context"):
            rel = r.get(key)
            if rel:
                p = os.path.join(args.data, rel)
                if os.path.isfile(p):
                    to_delete.append(p)

    print(f"file immagine da rimuovere: {len(to_delete)}")

    if not args.apply:
        print("\n[DRY-RUN] niente cancellato. Rilancia con --apply per eseguire.")
        return

    n = 0
    for p in to_delete:
        try:
            os.remove(p)
            n += 1
        except OSError as e:
            print(f"  [warn] non rimosso {p}: {e}")
    json.dump(keep_records, open(manifest_path, "w"), indent=2)
    print(f"\n── Fatto: rimossi {n} file, manifest ridotto a {len(keep_records)} record.")
    print("   labels.json e skips.json intatti.")


if __name__ == "__main__":
    main()
