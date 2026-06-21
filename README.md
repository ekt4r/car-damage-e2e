# car-damage-e2e
Production-ready car damage detection system.

| Model | Augmentations | Epochs | mAP | mAP50 | mAP75 |
|---|---|---:|---:|---:|---:|
| Faster R-CNN MobileNetV3 FPN | None | 5 | 0.4318 | 0.6307 | 0.4730 |
| Faster R-CNN MobileNetV3 FPN | Albumentations | 20 | 0.4681 | 0.6495 | 0.5060 |

## Commands

```bash
make validate
make debug
make train
make eval
```

## API

Run the FastAPI backend:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

For phone camera access, run the HTTPS backend:

```bash
make api-https
```

Open `http://127.0.0.1:8000/docs` and upload a car photo to `/predict`, or use curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict?score_threshold=0.5" \
  -F "file=@/path/to/car.jpg"
```

Open `http://127.0.0.1:8000/camera` for live camera detection in the browser.
On a phone, open `https://<your-computer-lan-ip>:8443/camera`; mobile browsers usually
require HTTPS before they allow camera access.

Environment overrides:

```bash
MODEL_CHECKPOINT=outputs/fasterrcnn_mobilenet_baseline/best_albumentations.pth
MODEL_CONFIG=configs/train.yaml
MODEL_DEVICE=auto
```
