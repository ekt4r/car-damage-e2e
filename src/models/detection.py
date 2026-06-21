import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(cfg):
    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"].get("pretrained", True)
    backbone = cfg["model"].get("backbone", "mobilenet_v3_large_fpn")
    min_size = cfg["model"].get("min_size", 800)
    max_size = cfg["model"].get("max_size", 1333)

    if backbone == "mobilenet_v3_large_fpn":
        weights = (
            torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            if pretrained
            else None
        )

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
        )

    elif backbone == "resnet50_fpn":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model