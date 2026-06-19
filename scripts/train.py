import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T

from src.data.dataset import CarDDDataset
from src.models.detection import build_model
from src.training.utils import collate_fn


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def move_targets_to_device(targets, device):
    return [
        {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in target.items()
        }
        for target in targets
    ]


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()

    running_loss = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        step += 1

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{running_loss / step:.4f}",
        )

    return running_loss / step


@torch.no_grad()
def validate_loss(model, loader, device, epoch, epochs):
    # TorchVision detection models return losses only in train mode.
    # We use no_grad(), so weights are not updated.
    model.train()

    running_loss = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Valid {epoch + 1}/{epochs}", leave=True)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

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


def maybe_subset(dataset, max_samples):
    if max_samples is None:
        return dataset
    return Subset(dataset, range(min(max_samples, len(dataset))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["train_split"],
        transforms=transforms,
    )

    val_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["val_split"],
        transforms=transforms,
    )

    train_dataset = maybe_subset(train_dataset, cfg["data"].get("max_train_samples"))
    val_dataset = maybe_subset(val_dataset, cfg["data"].get("max_val_samples"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
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
        pretrained=cfg["model"]["pretrained"],
    )
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    best_val_loss = float("inf")
    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, epochs)
        val_loss = validate_loss(model, val_loader, device, epoch, epochs)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
        }

        torch.save(checkpoint, output_dir / "last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"Saved best checkpoint: val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()