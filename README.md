# car-damage-e2e
Production-ready car damage detection system.

| Model | Augmentations | Epochs | mAP | mAP50 | mAP75 |
|---|---|---:|---:|---:|---:|
| Faster R-CNN MobileNetV3 FPN | None | 5 | 0.4318 | 0.6307 | 0.4730 |
| Faster R-CNN MobileNetV3 FPN | Albumentations | 20 | 0.4681 | 0.6495 | 0.5060 |
| Faster R-CNN ResNet50 FPN + EMA | Albumentations | 15/20 | 0.4720 | 0.6707 | 0.5008 |
| Faster R-CNN MobileNetV3 FPN + custom anchors | Albumentations | TBD | TBD | TBD | TBD |
| RetinaNet ResNet50 FPN + custom anchors | Albumentations | TBD | TBD | TBD | TBD |
| FCOS ResNet50 FPN | Albumentations | TBD | TBD | TBD | TBD |

Best ResNet50 + EMA checkpoint at epoch 15: val_loss=0.3051, mAR100=0.6209.
Per-class AP: dent=0.2016, scratch=0.2345, crack=0.1415, glass shatter=0.8501,
lamp broken=0.5777, tire flat=0.7555.

## Commands

```bash
make validate
make debug
make train
make eval
make predict
```

Generate GT vs prediction visualizations:

```bash
python scripts/predict.py \
  --config configs/experiments/resnet50_albu_ema.yaml \
  --checkpoint outputs/resnet50_albu_ema/best.pth \
  --data-dir data/raw/CarDD_COCO \
  --output-dir outputs/predictions \
  --num-samples 50 \
  --num-class-samples 50
```

Outputs are saved to `outputs/predictions/all`, `outputs/predictions/dent`,
`outputs/predictions/scratch`, and `outputs/predictions/crack`.

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
