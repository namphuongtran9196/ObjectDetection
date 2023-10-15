import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes

from configs.base import Config
from models import networks

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(image_path: str, transforms: A.Compose) -> torch.Tensor:
    """Preprocess image

    Args:
        image_path (str): path to image file
        transforms (A.Compose): albumentations transforms

    Returns:
        torch.Tensor: image tensor
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transforms(image=image)
    image = transformed["image"]
    image = image.div(255)
    return image


def main(opt: Config):
    if opt.classes_path is not None:
        with open(opt.classes_path, "r") as f:
            data = f.readlines()
        classes = [x.strip() for x in data]
        classes = [x.split()[-1] for x in classes]

    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, opt.model_type)(opt)
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(opt.model_type))

    logging.info("Loading checkpoint...")
    dict_checkpoint = torch.load(os.path.join(args.checkpoint_path), map_location=device)
    network.load_state_dict(dict_checkpoint["state_dict_network"])
    network.eval()

    logging.info("Preprocess image...")
    transforms = A.Compose(
        [
            A.Resize(opt.input_size_width, opt.input_size_height),
            ToTensorV2(),
        ]
    )
    logging.info("Start inference...")
    img_preprocessed = preprocess(args.image_path, transforms)
    img_preview = img_preprocessed.cpu() * 255
    img_preview = torch.tensor(img_preview, dtype=torch.uint8)

    with torch.no_grad():
        prediction = network([img_preprocessed.to(device)])[0]
        print(prediction)

    logging.info("Show result...")
    # Get classes name
    if opt.classes_path is not None:
        classes_name = [classes[i] for i in prediction["labels"][prediction["scores"] > 0.1].tolist()]
    else:
        classes_name = ["Unknown" for _ in prediction["labels"][prediction["scores"] > 0.1].tolist()]
    # Show image
    plt.imshow(
        draw_bounding_boxes(
            img_preview,
            prediction["boxes"][prediction["scores"] > 0.1],
            classes_name,
            width=4,
        ).permute(1, 2, 0)
    )
    plt.show()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("-cls_path", "--classes_path", type=str, help="Path to classes file classes.txt", default=None)
    parser.add_argument("-cfg_path", "--config_path", type=str, help="Path to config file opt.log")
    parser.add_argument("-ckpt_path", "--checkpoint_path", type=str, help="Path to checkpoint file ckpt.pt")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if not args.gpu:
        device = torch.device("cpu")
    opt = Config()
    opt.load(args.config_path)
    opt.classes_path = args.classes_path
    main(opt)
