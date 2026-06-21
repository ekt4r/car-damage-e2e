import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.data.dataset import CarDDDataset
from src.data.transforms import (
    DetectionAlbumentations,
    get_train_transforms,
    get_valid_transforms,
)
from src.models.detection import build_model
from src.training.utils import collate_fn, get_device, set_seed
from src.training.optim import build_optimizer
from src.training.schedulers import build_scheduler
from src.training.ema import ModelEMA
from src.data.transforms import DetectionAlbumentations, get_train_transforms, get_valid_transforms


def move_targets_to_device(targets, device):
    return [
        {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in target.items()
        }
        for target in targets
    ]


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, epochs, ema=None):
    model.train()

    running_loss = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad()

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_value = loss.item()
        running_loss += loss_value
        step += 1

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{running_loss / step:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return running_loss / step


@torch.no_grad()
def validate_loss(model, loader, device, epoch, epochs):
    # TorchVision detection models return losses only in train mode.
    model.train()

    running_loss = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Valid {epoch + 1}/{epochs}", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        loss_value = loss.item()
        running_loss += loss_value
        step += 1

        pbar.set_postfix(
            val_loss=f"{loss_value:.4f}",
            avg_val_loss=f"{running_loss / step:.4f}",
        )

    return running_loss / step


@torch.no_grad()
def evaluate_map(model, loader, device, epoch, epochs):
    model.eval()

    metric = MeanAveragePrecision(class_metrics=True)

    pbar = tqdm(loader, desc=f"mAP {epoch + 1}/{epochs}", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)

        outputs = [
            {k: v.cpu() for k, v in output.items()}
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

    metrics = metric.compute()

    return {
        "map": float(metrics["map"]),
        "map_50": float(metrics["map_50"]),
        "map_75": float(metrics["map_75"]),
        "mar_100": float(metrics["mar_100"]),
    }


def maybe_subset(dataset, max_samples):
    if max_samples is None:
        return dataset
    return Subset(dataset, range(min(max_samples, len(dataset))))


def apply_overrides(cfg, args):
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["optimizer"]["weight_decay"] = args.weight_decay
    if args.momentum is not None:
        cfg["optimizer"]["momentum"] = args.momentum
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("optimizer", {})
    cfg.setdefault("early_stopping", {})

    cfg["training"].setdefault("seed", 42)
    cfg["training"].setdefault("device", "auto")
    cfg["training"].setdefault("num_workers", 0)
    cfg["model"].setdefault("pretrained", True)

    cfg = apply_overrides(cfg, args)

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

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(cfg["training"]["output_dir"])
    )
    cfg["training"]["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(yaml.dump(cfg, sort_keys=False))
    print("=" * 60)

    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    set_seed(cfg["training"]["seed"])
    epochs = cfg["training"]["epochs"]

    train_transforms = DetectionAlbumentations(get_train_transforms(cfg))
    valid_transforms = DetectionAlbumentations(get_valid_transforms(cfg))

    train_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["train_split"],
        transforms=train_transforms,
    )
    val_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["val_split"],
        transforms=valid_transforms,
    )

    train_dataset = maybe_subset(train_dataset, cfg["data"].get("max_train_samples"))
    val_dataset = maybe_subset(val_dataset, cfg["data"].get("max_val_samples"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )

    model = build_model(cfg)
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    ema = None
    if cfg.get("ema", {}).get("enabled", False):
        ema = ModelEMA(model, decay=cfg["ema"]["decay"])

    best_map = -1.0
    patience = cfg.get("early_stopping", {}).get("patience")
    min_delta = cfg.get("early_stopping", {}).get("min_delta", 0.0)
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, epochs
        )
        val_loss = validate_loss(model, val_loader, device, epoch, epochs)
        val_metrics = evaluate_map(model, val_loader, device, epoch, epochs)
        val_map = val_metrics["map"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mAP={val_metrics['map']:.4f} | "
            f"mAP50={val_metrics['map_50']:.4f} | "
            f"mAP75={val_metrics['map_75']:.4f} | "
            f"mAR100={val_metrics['mar_100']:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_map": val_metrics["map"],
            "val_map_50": val_metrics["map_50"],
            "val_map_75": val_metrics["map_75"],
            "val_mar_100": val_metrics["mar_100"],
            "config": cfg,
        }

        torch.save(checkpoint, output_dir / "last.pth")

        if val_map > best_map + min_delta:
            best_map = val_map
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"Saved best checkpoint: mAP={best_map:.4f}")
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"Early stopping: no mAP improvement for "
                f"{patience} epoch(s). Best mAP={best_map:.4f}"
            )
            break

        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
