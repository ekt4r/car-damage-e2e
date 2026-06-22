import pytest
import torch

from src.data.dataset import CarDDDataset


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/CarDD_COCO").exists(),
    reason="CarDD_COCO dataset is not available locally",
)
def test_cardd_dataset_sample():
    dataset = CarDDDataset(
        data_dir="data/raw/CarDD_COCO",
        split="train",
        transforms=None,
    )

    image, target = dataset[0]

    assert image.mode == "RGB"
    assert "boxes" in target
    assert "labels" in target

    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64

    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4

    if len(target["boxes"]) > 0:
        boxes = target["boxes"]
        assert torch.all(boxes[:, 2] > boxes[:, 0])
        assert torch.all(boxes[:, 3] > boxes[:, 1])