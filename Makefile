.PHONY: validate debug train eval api api-https

DATA_DIR ?= data/raw/CarDD_COCO
OUTPUT_DIR ?= outputs/fasterrcnn_mobilenet_baseline
CHECKPOINT ?= $(OUTPUT_DIR)/best_albumentations.pth
HOST ?= 0.0.0.0
PORT ?= 8000
HTTPS_PORT ?= 8443
SSL_CERTFILE ?= certs/car-damage-local.crt
SSL_KEYFILE ?= certs/car-damage-local.key

validate:
	python scripts/validate_data.py --data-dir $(DATA_DIR)

debug:
	python scripts/train.py --config configs/debug_cpu.yaml --data-dir $(DATA_DIR) --output-dir outputs/debug_cpu

train:
	python scripts/train.py --config configs/train.yaml --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

eval:
	python scripts/evaluate.py --config configs/train.yaml --checkpoint $(OUTPUT_DIR)/best.pth --data-dir $(DATA_DIR)

api:
	MODEL_CHECKPOINT=$(CHECKPOINT) uvicorn src.api.main:app --host $(HOST) --port $(PORT)

api-https:
	MODEL_CHECKPOINT=$(CHECKPOINT) uvicorn src.api.main:app --host $(HOST) --port $(HTTPS_PORT) --ssl-certfile $(SSL_CERTFILE) --ssl-keyfile $(SSL_KEYFILE)
