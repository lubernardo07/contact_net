"""
generate_contact_crops.py — genera i crop "punta strumento" per l'annotazione di contatto
==========================================================================================
Gira sul SERVER (GPU). Usa lo STESSO percorso della pipeline per estrarre la punta dello
strumento (YOLO → maschera → extract_tool_tip → crop 224), così i crop annotati sono sulla
stessa distribuzione che il contact net vedrà in inferenza (niente mismatch train/inferenza).

Strategia di campionamento (per pescare i casi difficili "vicino ma non tocca"):
  - i frame ANNOTATI nel JSON BIOROB (tipicamente in contatto)
  - i frame a offset ±5/±10/±20 attorno a quelli (fasi di avvicinamento/ritrazione)
  - qualche frame uniforme di riempimento
Per ogni frame campionato, per ogni strumento YOLO → crop attorno alla punta.

Output (cartella --out):
  crops/<id>.jpg       crop RGB 224×224 attorno alla punta (input del classificatore)
  context/<id>.jpg     frame intero ridotto con punta+box marcati (per annotare a occhio)
  manifest.json        lista record {id, crop, context, video, frame_idx, tool_type, tip}

Uso:
  python generate_contact_crops.py --data_root data --split train \
      --out ../contact_net/contact_data --frames_per_video 10 --max_videos 300
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

# Moduli della pipeline (cartella sorella ../tti_pipeline)
_HERE = os.path.dirname(os.path.abspath(__file__))
_TTI  = os.path.abspath(os.path.join(_HERE, "..", "tti_pipeline"))
sys.path.insert(0, _TTI)

from config import CFG                          # noqa: E402
from models.loader import load_models           # noqa: E402
from processing.yolo_utils import run_yolo       # noqa: E402
from pipeline_B import extract_tool_tip, crop_around_point  # noqa: E402
from pipeline_A import _box_iou                  # noqa: E402


def _annotated_frames(labels_dir: str, stem: str) -> list:
    """Indici dei frame annotati nel JSON BIOROB (se presente)."""
    jf = os.path.join(labels_dir, stem + ".json")
    if not os.path.isfile(jf):
        return []
    try:
        data = json.load(open(jf))
    except Exception:
        return []
    return sorted(int(k) for k in data.get("labels", {}).keys())


def _sample_indices(n_frames: int, ann: list, per_video: int) -> list:
    """Frame annotati + offset (transizioni) + riempimento uniforme, dedup e clamp."""
    idxs = set()
    for f in ann:
        for off in (0, -5, 5, -10, 10, -20, 20):
            g = f + off
            if 0 <= g < n_frames:
                idxs.add(g)
    # riempimento uniforme fino a per_video
    if n_frames > 0:
        k = max(0, per_video - len(idxs))
        if k > 0:
            for v in np.linspace(2, max(2, n_frames - 3), k):
                idxs.add(int(round(v)))
    return sorted(i for i in idxs if 0 <= i < n_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="contact_data")
    ap.add_argument("--frames_per_video", type=int, default=10)
    ap.add_argument("--max_videos", type=int, default=0, help="0 = tutti")
    ap.add_argument("--max_crops_per_video", type=int, default=0,
                    help="tetto di crop NUOVI per video (0 = nessun tetto). I frame vengono "
                         "mescolati (deterministico per video) così i crop tenuti sono sparsi nel video.")
    ap.add_argument("--crop_size", type=int, default=224)
    ap.add_argument("--zoom_size", type=int, default=384,
                    help="vista locale (nativa, non ingrandita) attorno alla punta, per annotare a occhio")
    ap.add_argument("--context_max", type=int, default=1280)
    ap.add_argument("--append", action="store_true",
                    help="aggiunge ai crop/manifest esistenti saltando gli id già presenti (tiene labels.json)")
    args = ap.parse_args()

    # Path assoluti PRIMA di cambiare cartella; poi entro in tti_pipeline così i
    # pesi relativi di config.py ("checkpoints/...") si risolvono correttamente.
    args.data_root = os.path.abspath(args.data_root)
    args.out       = os.path.abspath(args.out)
    os.chdir(_TTI)

    videos_dir = os.path.join(args.data_root, "videos", args.split)
    labels_dir = os.path.join(args.data_root, "labels", args.split)
    vids = sorted(f for f in os.listdir(videos_dir)
                  if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if args.max_videos:
        vids = vids[:args.max_videos]

    crops_dir = os.path.join(args.out, "crops")
    ctx_dir   = os.path.join(args.out, "context")
    zoom_dir  = os.path.join(args.out, "zoom")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(ctx_dir,   exist_ok=True)
    os.makedirs(zoom_dir,  exist_ok=True)

    manifest_path = os.path.join(args.out, "manifest.json")
    existing, existing_ids = [], set()
    if args.append and os.path.isfile(manifest_path):
        existing = json.load(open(manifest_path))
        existing_ids = {r["id"] for r in existing}
        print(f"── append: {len(existing)} record esistenti — salto gli id già presenti")

    print("── Caricamento modelli (serve solo YOLO)…")
    yolo, _predictor, _vit, _depth, _device = load_models()

    manifest = []
    for vi, vname in enumerate(vids):
        vpath = os.path.join(videos_dir, vname)
        stem  = os.path.splitext(vname)[0]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [skip] non apribile: {vname}")
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ann   = _annotated_frames(labels_dir, stem)
        idxs  = _sample_indices(n_frames, ann, args.frames_per_video)
        if args.max_crops_per_video:
            random.Random(stem).shuffle(idxs)   # crop sparsi nel video, deterministico per video

        n_before = len(manifest)
        n_new_video = 0
        for fidx in idxs:
            if args.max_crops_per_video and n_new_video >= args.max_crops_per_video:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            H, W = frame.shape[:2]
            dets = run_yolo(yolo, frame)
            # Fallback OOD (come il seeding in pipeline_A): su dataset dove YOLO etichetta
            # lo strumento col VERBO (classi 8-14) e non col tipo (0-7), le box-verbo
            # localizzano comunque lo strumento → generiamo il crop, deducendo il tipo da
            # tti_verb_to_tool. Si saltano le box-verbo già coperte da una box-strumento.
            verb2tool = CFG.get("tti_verb_to_tool", {})
            use_tti   = CFG.get("seed_tools_from_tti", True) and bool(verb2tool)
            tool_boxes = [d["box"] for d in dets if d["class"] in CFG["tool_classes"]]
            for di, det in enumerate(dets):
                cls = det["class"]
                if cls in CFG["tool_classes"]:
                    tool_type = CFG["class_names"].get(cls, str(cls))
                elif use_tti and cls in verb2tool:
                    box = det.get("box")
                    if box is not None and any(_box_iou(box, tb) > 0.3 for tb in tool_boxes):
                        continue                       # già coperto da una box-strumento
                    tool_type = CFG["class_names"].get(verb2tool[cls], str(verb2tool[cls]))
                else:
                    continue
                m = det.get("mask")
                if m is None:
                    continue
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.uint8)
                tip = extract_tool_tip(m)
                if tip is None:
                    continue
                crop, (x1, y1, x2, y2) = crop_around_point(frame, tip[0], tip[1], args.crop_size)
                if crop.size == 0:
                    continue

                rid = f"{stem}_f{fidx:06d}_t{di}"
                if rid in existing_ids:
                    continue
                cv2.imwrite(os.path.join(crops_dir, rid + ".jpg"), crop)

                # zoom: finestra più ampia attorno alla punta a risoluzione NATIVA
                # (non ingrandita → nitida). Serve solo per annotare a occhio.
                zoom, _ = crop_around_point(frame, tip[0], tip[1], args.zoom_size)
                if zoom.size:
                    cv2.imwrite(os.path.join(zoom_dir, rid + ".jpg"), zoom)

                ctx = frame.copy()
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(ctx, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
                s = args.context_max / max(H, W) if max(H, W) > args.context_max else 1.0
                if s != 1.0:
                    ctx = cv2.resize(ctx, (int(W * s), int(H * s)))
                cv2.imwrite(os.path.join(ctx_dir, rid + ".jpg"), ctx)

                manifest.append({
                    "id":         rid,
                    "crop":       f"crops/{rid}.jpg",
                    "zoom":       f"zoom/{rid}.jpg",
                    "context":    f"context/{rid}.jpg",
                    "video":      stem,
                    "frame_idx":  fidx,
                    "tool_type":  tool_type,
                    "tip":        [int(tip[0]), int(tip[1])],
                })
                n_new_video += 1
                if args.max_crops_per_video and n_new_video >= args.max_crops_per_video:
                    break
        cap.release()
        print(f"  [{vi+1}/{len(vids)}] {stem}: +{len(manifest)-n_before} crop  (tot {len(manifest)})")

    all_records = existing + manifest
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n── Fatto: +{len(manifest)} nuovi crop, {len(all_records)} totali in {args.out}/  (manifest.json)")


if __name__ == "__main__":
    main()
