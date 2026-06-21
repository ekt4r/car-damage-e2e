import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.models.detection import build_model
from src.training.utils import get_device


DEFAULT_CHECKPOINT_PATH = "outputs/fasterrcnn_mobilenet_baseline/best_albumentations.pth"
DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}


class Detection(BaseModel):
    label_id: int
    label: str
    score: float
    box: list[float] = Field(description="[x_min, y_min, x_max, y_max] in source pixels")


class PredictionResponse(BaseModel):
    filename: str
    width: int
    height: int
    detections: list[Detection]


class ModelBundle:
    def __init__(self, model: torch.nn.Module, device: torch.device, class_names: dict[int, str]):
        self.model = model
        self.device = device
        self.class_names = class_names


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {"model": {"num_classes": 7}, "training": {"device": "auto"}}

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg["model"].setdefault("num_classes", 7)
    cfg["training"].setdefault("device", "auto")
    return cfg


def _checkpoint_config(checkpoint: dict) -> dict:
    cfg = checkpoint.get("config")
    return cfg if isinstance(cfg, dict) else {}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "auto"
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "auto"
    return get_device(device_name)


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    checkpoint_path = Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT_PATH))
    config_path = Path(os.getenv("MODEL_CONFIG", DEFAULT_CONFIG_PATH))
    device_name = os.getenv("MODEL_DEVICE")

    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    file_cfg = _load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = _checkpoint_config(checkpoint)

    model_cfg = {**file_cfg.get("model", {}), **checkpoint_cfg.get("model", {})}
    if device_name is not None:
        requested_device = device_name
    else:
        requested_device = file_cfg.get("training", {}).get("device", "auto")

    device = _resolve_device(requested_device)
    model = build_model(num_classes=int(model_cfg.get("num_classes", 7)), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return ModelBundle(model=model, device=device, class_names=DEFAULT_CLASS_NAMES)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = torch.from_numpy(np.array(image))
    return tensor.permute(2, 0, 1).float() / 255.0


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


app = FastAPI(
    title="Car Damage Detection API",
    version="0.1.0",
)


def detect_image(
    image: Image.Image,
    score_threshold: float,
    max_detections: int,
) -> list[Detection]:
    image_tensor = _image_to_tensor(image)

    bundle = get_model_bundle()
    with torch.inference_mode():
        output = bundle.model([image_tensor.to(bundle.device)])[0]

    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep][:max_detections]
    labels = labels[keep][:max_detections]
    scores = scores[keep][:max_detections]

    return [
        Detection(
            label_id=int(label),
            label=bundle.class_names.get(int(label), f"class {int(label)}"),
            score=round(float(score), 6),
            box=[round(float(value), 2) for value in box.tolist()],
        )
        for box, label, score in zip(boxes, labels, scores)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    bundle = get_model_bundle()
    return {
        "status": "ok",
        "device": str(bundle.device),
    }


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/camera")


@app.get("/camera", response_class=HTMLResponse)
def camera() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Car Damage Live Detection</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #1b1d22;
      --line: #333842;
      --text: #f2f4f8;
      --muted: #a7adba;
      --accent: #25d07d;
      --warn: #ffce5c;
      --danger: #ff6363;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    header, footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      border-color: var(--line);
      background: #15171b;
    }
    header { border-bottom: 1px solid var(--line); }
    footer { border-top: 1px solid var(--line); flex-wrap: wrap; }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
    }
    .stage {
      position: relative;
      min-height: 0;
      display: grid;
      place-items: center;
      overflow: hidden;
      background: #07080a;
    }
    video, #overlay {
      width: min(100vw, calc(100vh * 1.5));
      max-height: calc(100vh - 122px);
      aspect-ratio: 3 / 2;
      object-fit: contain;
    }
    #overlay {
      position: absolute;
      pointer-events: none;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    button {
      min-width: 92px;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #242832;
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      border-color: #168d57;
      background: #157a4d;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    label {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    input[type="range"] { width: 150px; }
    select {
      height: 36px;
      max-width: min(58vw, 320px);
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #242832;
      color: var(--text);
      padding: 0 8px;
    }
    .status {
      color: var(--muted);
      font-size: 14px;
      min-width: 180px;
      text-align: right;
    }
    .status.ok { color: var(--accent); }
    .status.warn { color: var(--warn); }
    .status.err { color: var(--danger); }
    @media (max-width: 700px) {
      header, footer { align-items: stretch; }
      header { flex-direction: column; }
      .status { text-align: left; min-width: 0; }
      .controls { width: 100%; }
      button { flex: 1 1 112px; }
      label { flex: 1 1 190px; }
      select { width: 100%; max-width: none; }
      video, #overlay {
        width: 100vw;
        max-height: calc(100vh - 188px);
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Car Damage Live Detection</h1>
      <div id="status" class="status">idle</div>
    </header>
    <section class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <canvas id="capture" hidden></canvas>
    </section>
    <footer>
      <div class="controls">
        <button id="start" class="primary">Start</button>
        <button id="stop" disabled>Stop</button>
        <select id="cameraSelect" aria-label="Camera"></select>
      </div>
      <div class="controls">
        <label>Threshold <input id="threshold" type="range" min="0.05" max="0.95" step="0.05" value="0.45"><span id="thresholdValue">0.45</span></label>
        <label>FPS <input id="fps" type="range" min="0.2" max="2" step="0.2" value="0.8"><span id="fpsValue">0.8</span></label>
      </div>
    </footer>
  </main>
  <script>
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const capture = document.getElementById("capture");
    const statusEl = document.getElementById("status");
    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    const cameraSelect = document.getElementById("cameraSelect");
    const threshold = document.getElementById("threshold");
    const thresholdValue = document.getElementById("thresholdValue");
    const fps = document.getElementById("fps");
    const fpsValue = document.getElementById("fpsValue");

    let stream = null;
    let loopTimer = null;
    let busy = false;

    function setStatus(text, kind = "") {
      statusEl.textContent = text;
      statusEl.className = `status ${kind}`;
    }

    function syncLabels() {
      thresholdValue.textContent = Number(threshold.value).toFixed(2);
      fpsValue.textContent = Number(fps.value).toFixed(1);
    }

    function checkSecureContext() {
      if (!window.isSecureContext) {
        setStatus("open HTTPS URL for camera", "err");
      }
    }

    async function loadCameras() {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      cameraSelect.innerHTML = "";
      cameras.forEach((camera, index) => {
        const option = document.createElement("option");
        option.value = camera.deviceId;
        option.textContent = camera.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });
    }

    async function startCamera() {
      stopCamera();
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("camera unavailable", "err");
        return;
      }

      const selectedDevice = cameraSelect.value;
      const videoConstraints = selectedDevice
        ? { deviceId: { exact: selectedDevice } }
        : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } };

      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
        video.srcObject = stream;
        await video.play();
        await loadCameras();
        startButton.disabled = true;
        stopButton.disabled = false;
        setStatus("running", "ok");
        scheduleNext();
      } catch (error) {
        setStatus(error.name || "camera error", "err");
      }
    }

    function stopCamera() {
      if (loopTimer) {
        clearTimeout(loopTimer);
        loopTimer = null;
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      busy = false;
      startButton.disabled = false;
      stopButton.disabled = true;
      clearOverlay();
      setStatus("stopped", "warn");
    }

    function scheduleNext() {
      const delay = Math.round(1000 / Number(fps.value));
      loopTimer = setTimeout(runDetection, delay);
    }

    function resizeCanvases() {
      const width = video.videoWidth || 960;
      const height = video.videoHeight || 640;
      const aspectRatio = `${width} / ${height}`;
      video.style.aspectRatio = aspectRatio;
      overlay.style.aspectRatio = aspectRatio;
      overlay.width = width;
      overlay.height = height;

      const targetWidth = Math.min(720, width);
      const targetHeight = Math.round(height * (targetWidth / width));
      capture.width = targetWidth;
      capture.height = targetHeight;
    }

    function clearOverlay() {
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    function drawDetections(result) {
      resizeCanvases();
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      const scaleX = overlay.width / result.width;
      const scaleY = overlay.height / result.height;
      ctx.lineWidth = Math.max(3, Math.round(overlay.width / 360));
      ctx.font = `${Math.max(16, Math.round(overlay.width / 46))}px system-ui, sans-serif`;
      ctx.textBaseline = "top";

      result.detections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.box;
        const x = x1 * scaleX;
        const y = y1 * scaleY;
        const w = (x2 - x1) * scaleX;
        const h = (y2 - y1) * scaleY;
        const text = `${detection.label} ${Math.round(detection.score * 100)}%`;
        const textWidth = ctx.measureText(text).width + 12;
        const textHeight = Math.max(24, Math.round(overlay.width / 32));

        ctx.strokeStyle = "#25d07d";
        ctx.fillStyle = "rgba(16, 17, 20, 0.82)";
        ctx.strokeRect(x, y, w, h);
        ctx.fillRect(x, Math.max(0, y - textHeight), textWidth, textHeight);
        ctx.fillStyle = "#f2f4f8";
        ctx.fillText(text, x + 6, Math.max(0, y - textHeight) + 4);
      });
      setStatus(`${result.detections.length} detections`, result.detections.length ? "ok" : "warn");
    }

    async function runDetection() {
      if (!stream || busy || video.readyState < 2) {
        scheduleNext();
        return;
      }

      busy = true;
      resizeCanvases();
      capture.getContext("2d").drawImage(video, 0, 0, capture.width, capture.height);

      capture.toBlob(async (blob) => {
        try {
          const body = new FormData();
          body.append("file", blob, "frame.jpg");
          const url = `/predict?score_threshold=${threshold.value}&max_detections=12`;
          const response = await fetch(url, { method: "POST", body });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          drawDetections(await response.json());
        } catch (error) {
          setStatus(error.message || "request error", "err");
        } finally {
          busy = false;
          if (stream) scheduleNext();
        }
      }, "image/jpeg", 0.78);
    }

    threshold.addEventListener("input", syncLabels);
    fps.addEventListener("input", syncLabels);
    startButton.addEventListener("click", startCamera);
    stopButton.addEventListener("click", stopCamera);
    cameraSelect.addEventListener("change", startCamera);
    video.addEventListener("loadedmetadata", resizeCanvases);
    window.addEventListener("resize", resizeCanvases);
    syncLabels();
    checkSecureContext();
    loadCameras().catch(() => {});
  </script>
</body>
</html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Car image")],
    score_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
    max_detections: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PredictionResponse:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    image = _read_image(await file.read())
    detections = detect_image(
        image=image,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    return PredictionResponse(
        filename=file.filename or "uploaded-image",
        width=image.width,
        height=image.height,
        detections=detections,
    )
