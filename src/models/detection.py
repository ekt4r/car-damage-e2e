import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    else:
        weights = None

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model