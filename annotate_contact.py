"""
annotate_contact.py — tool locale per etichettare i crop come contatto / non-contatto
=======================================================================================
Mostra il crop (ingrandito) e il frame di contesto con la punta marcata. Etichetti con un
tasto. Salva un labels.json incrementale (resume automatico: riparte dove avevi lasciato).

Tasti:
  c  → CONTATTO          (lo strumento tocca il tessuto)
  n  → NO CONTATTO       (vicino o lontano, ma non tocca)
  s  → SKIP              (ambiguo / crop inutile)
  z  → INDIETRO          (torna al precedente per correggere)
  q  → SALVA ED ESCI

I label vengono salvati ad ogni pressione (niente perdita di lavoro). Gli SKIP non
finiscono in labels.json ma vengono ricordati per non riproporli.

Uso:
  python annotate_contact.py --data contact_data
"""

import argparse
import json
import os
import tkinter as tk

from PIL import Image, ImageTk


class Annotator:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = data_dir
        self.manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
        self.labels_path = os.path.join(data_dir, "labels.json")
        self.skips_path  = os.path.join(data_dir, "skips.json")
        self.labels = json.load(open(self.labels_path)) if os.path.isfile(self.labels_path) else {}
        self.skips  = set(json.load(open(self.skips_path))) if os.path.isfile(self.skips_path) else set()

        # primo elemento non ancora etichettato/skippato
        self.idx = 0
        self._advance_to_unlabeled()
        self._photo = []  # refs per evitare GC

        root.title("Annotazione contatto")
        root.configure(bg="#222")
        self.info = tk.Label(root, font=("Consolas", 13), fg="#eee", bg="#222", justify="left")
        self.info.pack(pady=6)
        self.imgrow = tk.Frame(root, bg="#222")
        self.imgrow.pack()
        self.lbl_ctx  = tk.Label(self.imgrow, bg="#222")
        self.lbl_ctx.pack(side="left", padx=8)
        self.lbl_crop = tk.Label(self.imgrow, bg="#222")
        self.lbl_crop.pack(side="left", padx=8)
        self.help = tk.Label(
            root, bg="#222", fg="#9cf", font=("Consolas", 12),
            text="[c] contatto   [n] no-contatto   [s] skip   [z] indietro   [q] salva ed esci",
        )
        self.help.pack(pady=8)

        root.bind("c", lambda e: self.label("contact"))
        root.bind("n", lambda e: self.label("no_contact"))
        root.bind("s", lambda e: self.skip())
        root.bind("z", lambda e: self.back())
        root.bind("q", lambda e: self.quit())
        self.show()

    # ── navigazione ──────────────────────────────────────────────────────────
    def _is_done(self, i):
        rid = self.manifest[i]["id"]
        return rid in self.labels or rid in self.skips

    def _advance_to_unlabeled(self):
        while self.idx < len(self.manifest) and self._is_done(self.idx):
            self.idx += 1

    def _load_img(self, relpath, maxside):
        path = os.path.join(self.data_dir, relpath)
        if not os.path.isfile(path):
            return None
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = maxside / max(w, h)
        if s != 1.0:
            im = im.resize((int(w * s), int(h * s)))
        ph = ImageTk.PhotoImage(im)
        self._photo.append(ph)
        return ph

    def show(self):
        self._photo.clear()
        if self.idx >= len(self.manifest):
            self.info.config(text="── Tutto annotato! Premi [q] per salvare ed uscire.")
            self.lbl_ctx.config(image="")
            self.lbl_crop.config(image="")
            return
        rec = self.manifest[self.idx]
        n_c = sum(1 for v in self.labels.values() if v == "contact")
        n_n = sum(1 for v in self.labels.values() if v == "no_contact")
        self.info.config(text=(
            f"  {self.idx+1}/{len(self.manifest)}   "
            f"etichettati: {len(self.labels)}  (contatto {n_c} / no {n_n})  skip {len(self.skips)}\n"
            f"  video: {rec['video']}   frame: {rec['frame_idx']}   tool: {rec['tool_type']}"
        ))
        ctx  = self._load_img(rec["context"], 560)
        crop = self._load_img(rec["crop"], 360)
        self.lbl_ctx.config(image=ctx if ctx else "")
        self.lbl_crop.config(image=crop if crop else "")

    # ── azioni ───────────────────────────────────────────────────────────────
    def label(self, value):
        if self.idx >= len(self.manifest):
            return
        rid = self.manifest[self.idx]["id"]
        self.labels[rid] = value
        self.skips.discard(rid)
        self._save()
        self.idx += 1
        self._advance_to_unlabeled()
        self.show()

    def skip(self):
        if self.idx >= len(self.manifest):
            return
        rid = self.manifest[self.idx]["id"]
        self.skips.add(rid)
        self.labels.pop(rid, None)
        self._save()
        self.idx += 1
        self._advance_to_unlabeled()
        self.show()

    def back(self):
        j = self.idx - 1
        while j >= 0 and not self._is_done(j):
            j -= 1
        if j >= 0:
            rid = self.manifest[j]["id"]
            self.labels.pop(rid, None)
            self.skips.discard(rid)
            self.idx = j
            self._save()
            self.show()

    def _save(self):
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)
        with open(self.skips_path, "w") as f:
            json.dump(sorted(self.skips), f, indent=2)

    def quit(self):
        self._save()
        n_c = sum(1 for v in self.labels.values() if v == "contact")
        n_n = sum(1 for v in self.labels.values() if v == "no_contact")
        print(f"Salvato {self.labels_path}: {len(self.labels)} label (contatto {n_c} / no {n_n}), skip {len(self.skips)}")
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="contact_data", help="cartella con manifest.json + crops/ + context/")
    args = ap.parse_args()
    root = tk.Tk()
    Annotator(root, args.data)
    root.mainloop()


if __name__ == "__main__":
    main()
