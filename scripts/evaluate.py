import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.data.dataset import CarDDDataset
from src.data.coco import read_coco_json, get_split_paths
from src.models.detection import build_model
from src.training.utils import collate_fn, get_device
from src.data.transforms import DetectionAlbumentations, get_valid_transforms


DEFAULT_CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_class_names(data_dir, val_split):
    annotations_path = get_split_paths(data_dir, val_split)["annotations_path"]
    if not annotations_path.exists():
        return DEFAULT_CLASS_NAMES

    coco = read_coco_json(annotations_path)
    categories = coco.get("categories", [])
    if not categories:
        return DEFAULT_CLASS_NAMES

    return {
        int(category["id"]): category["name"]
        for category in categories
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    metric = MeanAveragePrecision(class_metrics=True)

    pbar = tqdm(loader, desc="Evaluating", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]

        outputs = model(images)

        outputs = [
            {
                k: v.cpu()
                for k, v in output.items()
            }
            for output in outputs
        ]

        targets = [
            {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            }
            for target in targets
        ]

        metric.update(outputs, targets)

    return metric.compute()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml",)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("training", {})

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        cfg["data"]["data_dir"] = str(data_dir)
    elif cfg["data"].get("data_dir") is not None:
        data_dir = Path(cfg["data"]["data_dir"])
    else:
        raise ValueError(
            "Dataset directory is not specified. "
            "Pass --data-dir or set data.data_dir in the config."
        )
    
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers

    if args.device is not None:
        cfg["training"]["device"] = args.device

    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples

    device = get_device(cfg["training"].get("device", "auto"))
    print(f"Using device: {device}")

    transforms = DetectionAlbumentations(get_valid_transforms())

    val_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["val_split"],
        transforms=transforms,
    )

    max_val_samples = cfg["data"].get("max_val_samples")

    if max_val_samples is not None:
        val_dataset = Subset(
            val_dataset,
            range(min(max_val_samples, len(val_dataset))),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    )

    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.to(device)

    metrics = evaluate(
        model,
        val_loader,
        device,
    )
    class_names = load_class_names(data_dir, cfg["data"]["val_split"])

    print("\n========== Evaluation ==========")
    print(f"mAP        : {metrics['map']:.4f}")
    print(f"mAP50      : {metrics['map_50']:.4f}")
    print(f"mAP75      : {metrics['map_75']:.4f}")
    print(f"mAR100     : {metrics['mar_100']:.4f}")

    if "classes" in metrics and "map_per_class" in metrics:
        print("\nPer-class AP")

        for cls, ap in zip(metrics["classes"], metrics["map_per_class"]):
            class_id = int(cls)
            class_name = class_names.get(class_id, f"class {class_id}")
            print(f"{class_name:14s} ({class_id:2d}): {float(ap):.4f}")


if __name__ == "__main__":
    main()
