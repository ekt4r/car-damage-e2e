import pytest
import torch
from pathlib import Path

from src.data.dataset import CarDDDataset
from src.data.transforms import (
    DetectionAlbumentations,
    get_train_transforms,
    get_valid_transforms,
)


@pytest.mark.skipif(
    not Path("data/raw/CarDD_COCO").exists(),
    reason="CarDD_COCO dataset is not available locally",
)
def test_train_transforms_keep_valid_boxes():
    cfg = {
        "augmentations": {
            "train": {
                "horizontal_flip": True,
                "brightness_contrast": True,
                "motion_blur": True,
                "gauss_noise": True,
                "affine": False,
            }
        }
    }

    transforms = DetectionAlbumentations(get_train_transforms(cfg))

    dataset = CarDDDataset(
        data_dir="data/raw/CarDD_COCO",
        split="train",
        transforms=transforms,
    )

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3
    assert image.dtype == torch.float32

    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4

    if len(target["boxes"]) > 0:
        boxes = target["boxes"]
        assert torch.all(torch.isfinite(boxes))
        assert torch.all(boxes[:, 2] > boxes[:, 0])
        assert torch.all(boxes[:, 3] > boxes[:, 1])
        assert torch.all(target["area"] > 0)


@pytest.mark.skipif(
    not Path("data/raw/CarDD_COCO").exists(),
    reason="CarDD_COCO dataset is not available locally",
)
def test_valid_transforms_return_tensor():
    cfg = {}

    transforms = DetectionAlbumentations(get_valid_transforms(cfg))

    dataset = CarDDDataset(
        data_dir="data/raw/CarDD_COCO",
        split="val",
        transforms=transforms,
    )

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert target["boxes"].shape[1] == 4