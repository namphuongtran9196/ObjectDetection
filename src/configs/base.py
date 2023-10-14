import logging
import os
from abc import ABC, abstractmethod
from typing import List


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, opt: str):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(opt.checkpoint_dir), exist_ok=True)
        out_opt = os.path.join(opt.checkpoint_dir, "opt.log")
        with open(out_opt, "w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logging.info(message)

    def load(self, opt_path: str):
        def decode_value(value: str):
            value = value.strip()
            if "." in value and value.replace(".", "").isdigit():
                value = float(value)
            elif value.isdigit():
                value = int(value)
            elif value == "True":
                value = True
            elif value == "False":
                value = False
            elif value == "None":
                value = None
            elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value

        with open(opt_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = data[i].split(":")[0].strip(), data[i].split(":")[1].strip()
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [decode_value(x) for x in value]
                else:
                    value = decode_value(value)

                data_dict[key] = value
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # Training settings
        self.num_epochs: int = 50
        self.checkpoint_dir: str = "checkpoints"
        self.save_all_states: bool = True
        self.save_best_val: bool = True
        self.max_to_keep: int = 1
        self.save_freq: int = 4000
        self.batch_size: int = 2

        # Resume training
        self.resume: bool = False
        self.opt_path: str = None

        # Optimizer
        self.optimizer: str = "SGD"
        self.learning_rate: float = 0.01
        self.momentum: float = 0.9
        self.weight_decay: float = 1e-4
        self.nesterov: bool = True

        # learning rate scheduler
        self.warmup_factor: float = 0.0001  # None for no warmup
        self.warmup_iters: int = 1000  # should be less than dataloader size

        # Dataset
        self.classes_path: str = "data/classes.txt"
        self.data_root: str = "data/bakery.v1i.coo"
        self.data_format: str = "CocoDataset"  # [CocoDataset, YoLoDataset]

        # Image settings
        self.input_size_width: int = 640
        self.input_size_height: int = 640
        # the probability of each augmentation
        self.horizontal_flip: bool = 0.3
        self.vertical_flip: bool = 0.3
        self.brightness_contrast: bool = 0.1
        self.color_jitter: bool = 0.1

        # Model
        self.num_classes: int = 4
        self.model_type: str = "faster_rcnn"  # [faster_rcnn]
        self.trainer_type: str = "FasterRCNNTrainer"  # [FasterRCNNTrainer]
        self.backbone_name: str = "fasterrcnn_mobilenet_v3_large_fpn"  # [fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn]

        for key, value in kwargs.items():
            setattr(self, key, value)
