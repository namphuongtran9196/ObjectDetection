import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import nn, optim

from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import networks
from trainer import Trainer
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt: Config):
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, opt.model_type)(opt)
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(opt.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    opt.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(opt.checkpoint_dir),
        opt.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    opt.save(opt)

    # Build dataset
    train_ds, test_ds = build_train_test_dataset(opt)

    logging.info("Initializing trainer...")
    trainer = Trainer(
        network=network,
        criterion=None,
        log_dir=opt.checkpoint_dir,
    )

    logging.info("Start training...")
    # Build optimizer and criterion
    assert opt.optimizer in ["SGD"], "Optimizer {} is not implemented".format(opt.optimizer)
    optimizer = optim.SGD(
        params=trainer.network.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        nesterov=opt.nesterov,
        weight_decay=opt.weight_decay,
    )

    lr_scheduler = None
    if opt.warmup_factor is not None:
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=opt.warmup_factor, total_iters=opt.warmup_iters)

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=opt.save_freq,
        max_to_keep=opt.max_to_keep,
        save_best_val=opt.save_best_val,
        save_all_states=opt.save_all_states,
    )
    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    if opt.resume:
        trainer.load_all_states(opt.resume_path)
    trainer.fit(train_ds, opt.num_epochs, test_ds, callbacks=[ckpt_callback])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    opt: Config = get_options(args.config)
    if opt.resume and opt.opt_path is not None:
        resume = opt.resume
        resume_path = opt.resume_path
        opt.load(opt.opt_path)
        opt.resume = resume
        opt.resume_path = resume_path

    main(opt)