# contact_net — classificatore di contatto strumento-tessuto

Progetto sorella di `tti_pipeline`. Obiettivo: addestrare un classificatore binario
**contatto / non-contatto** sui crop "punta strumento", per superare il limite della
depth (che non distingue "tocca" da "complanare ma sospeso").

Motivazione (dall'analisi delle annotazioni BIOROB): le label esistenti sono di
*interazione* (quale strumento tocca quale tessuto), con pochissimi negativi
"vicino ma non tocca" (~38-150 in tutto). Per imparare l'ambiguità servono negativi
difficili → si annotano a mano, ma solo in modo **binario** (veloce).

## Flusso

1. **Generazione crop** (SERVER, GPU):
   ```
   python generate_contact_crops.py --data_root data --split train \
       --out ../contact_net/contact_data --frames_per_video 10 --max_videos 300
   ```
   Usa la stessa estrazione punta della pipeline (YOLO → maschera → `extract_tool_tip`
   → crop 224). Campiona attorno ai frame annotati (incl. avvicinamento/ritrazione) per
   pescare i casi difficili. Produce `crops/`, `context/`, `manifest.json`.

2. **Annotazione** (LOCALE, GUI): copia la cartella `contact_data/` in locale, poi
   ```
   python annotate_contact.py --data contact_data
   ```
   Tasti: `c` contatto · `n` no-contatto · `s` skip · `z` indietro · `q` salva ed esci.
   Resume automatico. Produce `labels.json` (e `skips.json`).

3. **Dataset + training** (prossimi script): `create_dataset_contact.py` + `train_contact.py`
   costruiscono il dataset da `manifest.json` + `labels.json` e addestrano un ResNet
   binario (RGB; depth come canale extra in ablation).

## Formato JSON

`manifest.json` (generazione):
```json
[{"id": "...", "crop": "crops/....jpg", "context": "context/....jpg",
  "video": "...", "frame_idx": 42, "tool_type": "Grasper", "tip": [311, 248]}]
```
`labels.json` (annotazione):
```json
{"<id>": "contact", "<id>": "no_contact"}
```
