import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.data.coco import get_default_data_dir, get_split_paths, read_coco_json
from src.data.dataset import CarDDDataset
from src.models.detection import build_model
from src.training.tta import predict_with_tta
from src.training.utils import get_device


DEFAULT_CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}
FOCUS_CLASSES = ("dent", "scratch", "crack")
GT_COLOR = (40, 190, 255)
PRED_COLOR = (255, 92, 92)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tta", {})
    cfg["training"].setdefault("device", "auto")
    cfg["tta"].setdefault("enabled", False)
    return cfg


def load_class_names(data_dir: Path, split: str) -> dict[int, str]:
    annotations_path = get_split_paths(data_dir, split)["annotations_path"]
    if not annotations_path.exists():
        return DEFAULT_CLASS_NAMES

    coco = read_coco_json(annotations_path)
    categories = coco.get("categories", [])
    if not categories:
        return DEFAULT_CLASS_NAMES

    return {int(category["id"]): category["name"] for category in categories}


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = torch.from_numpy(np.array(image))
    return tensor.permute(2, 0, 1).float() / 255.0


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "auto"
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "auto"
    return get_device(device_name)


def get_font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None,
    class_names: dict[int, str],
    color: tuple[int, int, int],
    title: str,
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = get_font(size=max(14, image.width // 55))
    line_width = max(2, image.width // 300)

    title_w, title_h = text_size(draw, title, font)
    draw.rectangle((0, 0, title_w + 14, title_h + 10), fill=(20, 20, 20))
    draw.text((7, 5), title, fill=(255, 255, 255), font=font)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        label_id = int(labels[idx])
        label = class_names.get(label_id, f"class {label_id}")
        if scores is None:
            caption = label
        else:
            caption = f"{label} {float(scores[idx]):.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_w, text_h = text_size(draw, caption, font)
        label_y = max(0, y1 - text_h - 8)
        draw.rectangle((x1, label_y, x1 + text_w + 10, label_y + text_h + 8), fill=color)
        draw.text((x1 + 5, label_y + 4), caption, fill=(0, 0, 0), font=font)

    return image


def make_side_by_side(
    image: Image.Image,
    target: dict,
    prediction: dict,
    class_names: dict[int, str],
    score_threshold: float,
    max_detections: int,
) -> Image.Image:
    keep = prediction["scores"].detach().cpu() >= score_threshold
    pred_boxes = prediction["boxes"].detach().cpu()[keep][:max_detections]
    pred_labels = prediction["labels"].detach().cpu()[keep][:max_detections]
    pred_scores = prediction["scores"].detach().cpu()[keep][:max_detections]

    gt_view = draw_boxes(
        image=image,
        boxes=target["boxes"].detach().cpu(),
        labels=target["labels"].detach().cpu(),
        scores=None,
        class_names=class_names,
        color=GT_COLOR,
        title="GT",
    )
    pred_view = draw_boxes(
        image=image,
        boxes=pred_boxes,
        labels=pred_labels,
        scores=pred_scores,
        class_names=class_names,
        color=PRED_COLOR,
        title=f"Pred >= {score_threshold:.2f}",
    )

    canvas = Image.new("RGB", (image.width * 2, image.height), color=(10, 10, 10))
    canvas.paste(gt_view, (0, 0))
    canvas.paste(pred_view, (image.width, 0))
    return canvas


def sample_indices_by_class(
    dataset: CarDDDataset,
    class_name_to_id: dict[str, int],
    num_samples: int,
) -> dict[str, list[int]]:
    selected = {class_name: [] for class_name in FOCUS_CLASSES}

    for idx, sample in enumerate(dataset.samples):
        labels = set(int(label) for label in sample["labels"].tolist())

        for class_name in FOCUS_CLASSES:
            class_id = class_name_to_id.get(class_name)
            if class_id in labels and len(selected[class_name]) < num_samples:
                selected[class_name].append(idx)

        if all(len(indices) >= num_samples for indices in selected.values()):
            break

    return selected


def safe_stem(path: Path, idx: int) -> str:
    return f"{idx:04d}_{path.stem}"


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tta_enabled: bool,
) -> dict:
    image_tensor = image_to_tensor(image).to(device)
    output = predict_with_tta(model, [image_tensor], enabled=tta_enabled)[0]
    return {key: value.detach().cpu() for key, value in output.items()}


def save_visualization(
    dataset: CarDDDataset,
    idx: int,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    class_names: dict[int, str],
    score_threshold: float,
    max_detections: int,
    tta_enabled: bool,
) -> None:
    image, target = dataset[idx]
    prediction = predict_one(
        model=model,
        image=image,
        device=device,
        tta_enabled=tta_enabled,
    )
    visualization = make_side_by_side(
        image=image,
        target=target,
        prediction=prediction,
        class_names=class_names,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_stem(target['path'], idx)}.jpg"
    visualization.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-class-samples", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or cfg["data"].get("data_dir") or get_default_data_dir())
    split = args.split or cfg["data"].get("val_split", "val")
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device or cfg["training"].get("device", "auto"))
    tta_enabled = bool(cfg.get("tta", {}).get("enabled", False)) and not args.disable_tta

    dataset = CarDDDataset(data_dir=data_dir, split=split, transforms=None)
    class_names = load_class_names(data_dir, split)
    class_name_to_id = {name: class_id for class_id, name in class_names.items()}

    model = build_model(cfg)
    load_checkpoint(model, Path(args.checkpoint), device)
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"TTA enabled: {tta_enabled}")
    print(f"Saving visualizations to: {output_dir}")

    general_indices = list(range(min(args.num_samples, len(dataset))))
    class_indices = sample_indices_by_class(
        dataset=dataset,
        class_name_to_id=class_name_to_id,
        num_samples=args.num_class_samples,
    )

    for idx in tqdm(general_indices, desc="Saving all"):
        save_visualization(
            dataset=dataset,
            idx=idx,
            output_dir=output_dir / "all",
            model=model,
            device=device,
            class_names=class_names,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            tta_enabled=tta_enabled,
        )

    for class_name, indices in class_indices.items():
        for idx in tqdm(indices, desc=f"Saving {class_name}"):
            save_visualization(
                dataset=dataset,
                idx=idx,
                output_dir=output_dir / class_name.replace(" ", "_"),
                model=model,
                device=device,
                class_names=class_names,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                tta_enabled=tta_enabled,
            )

        if len(indices) < args.num_class_samples:
            print(
                f"Only found {len(indices)} {class_name} samples "
                f"for requested {args.num_class_samples}."
            )


if __name__ == "__main__":
    main()
