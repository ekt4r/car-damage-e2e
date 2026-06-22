import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetHead
from torchvision.models.detection.rpn import RPNHead


def _build_anchor_generator(anchor_cfg):
    sizes = anchor_cfg["sizes"]
    aspect_ratios = anchor_cfg["aspect_ratios"]

    return AnchorGenerator(
        sizes=tuple(tuple(size) for size in sizes),
        aspect_ratios=tuple(tuple(ratio) for ratio in aspect_ratios),
    )


def _common_kwargs(model_cfg):
    return {
        "min_size": model_cfg.get("min_size", 800),
        "max_size": model_cfg.get("max_size", 1333),
    }


def _build_fasterrcnn(cfg):
    model_cfg = cfg["model"]
    num_classes = model_cfg["num_classes"]
    pretrained = model_cfg.get("pretrained", True)
    backbone = model_cfg.get("backbone", "mobilenet_v3_large_fpn")
    kwargs = _common_kwargs(model_cfg)
    kwargs["box_score_thresh"] = model_cfg.get("box_score_thresh", 0.05)
    kwargs["box_nms_thresh"] = model_cfg.get("box_nms_thresh", 0.5)

    anchor_cfg = model_cfg.get("anchors", {})
    custom_anchors = anchor_cfg.get("enabled", False)

    if backbone == "mobilenet_v3_large_fpn":
        weights = (
            torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        weights_backbone = (
            None if not pretrained else torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        )

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            **kwargs,
        )

    elif backbone == "resnet50_fpn":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        weights_backbone = None if not pretrained else torchvision.models.ResNet50_Weights.DEFAULT

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown Faster R-CNN backbone: {backbone}")

    if custom_anchors:
        anchor_generator = _build_anchor_generator(anchor_cfg)
        num_anchors_per_location = anchor_generator.num_anchors_per_location()
        if len(set(num_anchors_per_location)) != 1:
            raise ValueError(
                "Faster R-CNN requires the same number of anchors per location "
                "for every feature map."
            )
        model.rpn.anchor_generator = anchor_generator
        model.rpn.head = RPNHead(model.backbone.out_channels, num_anchors_per_location[0])

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def _build_retinanet(cfg):
    model_cfg = cfg["model"]
    backbone = model_cfg.get("backbone", "resnet50_fpn")
    if backbone != "resnet50_fpn":
        raise ValueError(f"Unknown RetinaNet backbone: {backbone}")

    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg["num_classes"]
    kwargs = _common_kwargs(model_cfg)
    kwargs["score_thresh"] = model_cfg.get("box_score_thresh", 0.05)
    kwargs["nms_thresh"] = model_cfg.get("box_nms_thresh", 0.5)

    anchor_cfg = model_cfg.get("anchors", {})
    custom_anchors = anchor_cfg.get("enabled", False)

    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=(
            torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        ),
        weights_backbone=None,
        num_classes=None if pretrained else num_classes,
        **kwargs,
    )

    if custom_anchors:
        anchor_generator = _build_anchor_generator(anchor_cfg)
        num_anchors_per_location = anchor_generator.num_anchors_per_location()
        if len(set(num_anchors_per_location)) != 1:
            raise ValueError(
                "RetinaNet requires the same number of anchors per location "
                "for every feature map."
            )
        model.anchor_generator = anchor_generator
        model.head = RetinaNetHead(
            model.backbone.out_channels,
            num_anchors_per_location[0],
            num_classes,
        )
    elif pretrained:
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            num_classes,
        )

    return model


def _build_fcos(cfg):
    model_cfg = cfg["model"]
    backbone = model_cfg.get("backbone", "resnet50_fpn")
    if backbone != "resnet50_fpn":
        raise ValueError(f"Unknown FCOS backbone: {backbone}")

    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg["num_classes"]
    kwargs = _common_kwargs(model_cfg)
    kwargs["score_thresh"] = model_cfg.get("box_score_thresh", 0.05)
    kwargs["nms_thresh"] = model_cfg.get("box_nms_thresh", 0.5)

    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights=(
            torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        ),
        weights_backbone=None,
        num_classes=None if pretrained else num_classes,
        **kwargs,
    )

    if pretrained:
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            num_classes,
        )

    return model


def build_model(cfg):
    model_name = cfg["model"].get("name", "fasterrcnn").lower()

    if model_name == "fasterrcnn":
        return _build_fasterrcnn(cfg)
    if model_name == "retinanet":
        return _build_retinanet(cfg)
    if model_name == "fcos":
        return _build_fcos(cfg)

    raise ValueError(f"Unknown model name: {model_name}")
