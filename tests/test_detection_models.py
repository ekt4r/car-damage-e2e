import torch

from src.models.detection import build_model


def _base_cfg(name, backbone="resnet50_fpn", anchors=None):
    model = {
        "name": name,
        "backbone": backbone,
        "num_classes": 7,
        "pretrained": False,
        "min_size": 128,
        "max_size": 256,
        "box_score_thresh": 0.05,
        "box_nms_thresh": 0.5,
    }
    if anchors is not None:
        model["anchors"] = anchors
    return {"model": model}


def _tiny_train_forward(model):
    model.train()
    images = [torch.rand(3, 256, 256)]
    targets = [
        {
            "boxes": torch.tensor([[30.0, 40.0, 160.0, 180.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]

    loss_dict = model(images, targets)

    assert loss_dict
    for loss in loss_dict.values():
        assert torch.isfinite(loss).all()


def test_build_fasterrcnn_mobilenet():
    model = build_model(_base_cfg("fasterrcnn", "mobilenet_v3_large_fpn"))

    _tiny_train_forward(model)


def test_build_fasterrcnn_resnet50():
    model = build_model(_base_cfg("fasterrcnn", "resnet50_fpn"))

    _tiny_train_forward(model)


def test_build_fasterrcnn_custom_anchors():
    anchors = {
        "enabled": True,
        "sizes": [[32], [64], [128], [256], [512]],
        "aspect_ratios": [
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ],
    }
    model = build_model(_base_cfg("fasterrcnn", "resnet50_fpn", anchors))

    _tiny_train_forward(model)


def test_build_retinanet_resnet50():
    anchors = {
        "enabled": True,
        "sizes": [[16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512], [256, 512, 768]],
        "aspect_ratios": [
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.25, 0.5, 1.0, 2.0, 4.0],
        ],
    }
    model = build_model(_base_cfg("retinanet", "resnet50_fpn", anchors))

    _tiny_train_forward(model)


def test_build_fcos_resnet50():
    model = build_model(_base_cfg("fcos", "resnet50_fpn"))

    _tiny_train_forward(model)
