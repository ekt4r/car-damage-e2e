from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models.detection import build_model
from src.data.dataset import CarDDDataset
from src.training.utils import collate_fn
from torch.utils.data import DataLoader
import torch
import argparse
import yaml
from pathlib import Path
from torchvision.transforms import v2 as T
from tqdm import tqdm


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["data_dir"])

    device = 'mps' #get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    model = build_model(num_classes=7, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()
    metric = MeanAveragePrecision()

    val_dataset = CarDDDataset(
        data_dir=data_dir,
        split=cfg["data"]["val_split"],
        transforms=transforms,
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            collate_fn=collate_fn,
        )

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            outputs = model(images)

            outputs = [
                {k: v.cpu() for k, v in out.items()}
                for out in outputs
            ]

            targets = [
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
                for target in targets
            ]

            metric.update(outputs, targets)

    result = metric.compute()
    print(result)

if __name__ == '__main__':
    main()