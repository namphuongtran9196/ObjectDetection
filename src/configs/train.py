from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        # Training settings
        self.num_epochs: int = 50
        self.checkpoint_dir: str = "checkpoints"
        self.batch_size: int = 32

        # Optimizer
        self.optimizer: str = "SGD"
        self.learning_rate: float = 0.01

        # learning rate scheduler
        self.warmup_factor: float = 0.0001  # None for no warmup
        self.warmup_iters: int = 1000  # should be less than dataloader size

        # Dataset
        self.data_root: str = "data/coco_yolo"
        self.data_format: str = "YoLoDataset"  # [CocoDataset, YoLoDataset]

        # Image settings
        self.input_size_width: int = 640
        self.input_size_height: int = 640

        # Model
        self.num_classes: int = 46
        self.model_type: str = "faster_rcnn"  # [faster_rcnn]
        self.trainer_type: str = "FasterRCNNTrainer"  # [FasterRCNNTrainer]
        self.backbone_name: str = "fasterrcnn_mobilenet_v3_large_fpn"  # [fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn]

        # Config name
        self.name = f"{self.model_type}_{self.backbone_name}"

        for key, value in kwargs.items():
            setattr(self, key, value)
