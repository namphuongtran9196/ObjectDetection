from torchvision import models

from configs.base import Config

from .faster_rcnn import build_faster_rcnn


def faster_rcnn(opt: Config) -> models.detection.FasterRCNN:
    """Build Faster RCNN model

    Args:
        num_classes (int): the number of classes
        backbone_name (str, optional): the name of the backbone model which is used in Faster RCNN. Defaults to "fasterrcnn_resnet50_fpn".
        the name of the backbone model can be one of the following:
                "fasterrcnn_resnet50_fpn"
                "fasterrcnn_resnet50_fpn_v2"
                "fasterrcnn_mobilenet_v3_large_fpn"
                "fasterrcnn_mobilenet_v3_large_320_fpn"

    Returns:
        models.detection.FasterRCNN: Faster RCNN model
    """

    return build_faster_rcnn(opt.backbone_name, pretrained=True, num_classes=opt.num_classes)
