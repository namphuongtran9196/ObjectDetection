from torchvision import models

# torch version: 2.0.1
__all__ = [
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
]


def build_faster_rcnn(backbone_name: str, pretrained: bool, num_classes: int, **kwargs) -> models.detection.FasterRCNN:
    """Build Faster RCNN model

    Args:
        backbone_name (str): the name of the backbone model which is used in Faster RCNN
        pretrained (bool): whether to load pretrained weights
        num_classes (int): the number of classes

    Returns:
        models.detection.FasterRCNN: Faster RCNN model
    """
    # check if the backbone_name is in __all__
    assert backbone_name in __all__, f"backbone_name must be in {__all__}"
    # get the backbone model
    backbone = getattr(models.detection, backbone_name)(pretrained=pretrained, **kwargs)
    # adjust the number of classes
    in_features = backbone.roi_heads.box_predictor.cls_score.in_features
    backbone.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return backbone
