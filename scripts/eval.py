import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import random

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import metrics, networks

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt: Config, confidence: float):
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

    logging.info("Build dataset...")
    # Get test_dataset
    opt.batch_size = 1
    _, test_ds = build_train_test_dataset(opt, val_split="test")
    test_loader = iter(test_ds)

    logging.info("Start evaluating...")
    ground_truth = []
    predictions = []

    for inputs, targets in test_loader:
        # add to ground truth
        gt = []
        target = [{k: v.clone().detach().to(device) for k, v in t.items()} for t in targets][0]
        boxes = target["boxes"].cpu().numpy().tolist()
        labels = target["labels"].cpu().numpy().tolist()
        for box, label in zip(boxes, labels):
            gt.append(box + [label])
        ground_truth.append(gt)

        # add to predictions
        preds = []
        inputs = list(img.to(device) for img in inputs)
        with torch.no_grad():
            prediction = network(input)[0]
            boxes = prediction["boxes"][prediction["scores"] > confidence].detach().cpu().numpy().tolist()
            scores = prediction["scores"][prediction["scores"] > confidence].detach().cpu().numpy().tolist()
            labels = prediction["labels"][prediction["scores"] > confidence].detach().cpu().numpy().tolist()
        for box, score, label in zip(boxes, scores, labels):
            preds.append(box + [label, score])
        predictions.append(preds)

    metrics.coco_mAP(ground_truth, predictions, num_classes=opt.num_classes, summarize=True)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg_path", "--config_path", type=str, help="Path to config file opt.log")
    parser.add_argument("-ckpt_path", "--checkpoint_path", type=str, help="Path to checkpoint file ckpt.pt")
    parser.add_argument("-conf", "--confidence", type=float, help="Confidence threshold", default=0.01)
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if not args.gpu:
        device = torch.device("cpu")
    opt = Config()
    opt.load(args.config_path)
    main(opt, args.confidence)
