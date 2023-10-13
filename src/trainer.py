from typing import Dict, Union

import torch
from torch import Tensor
from torchvision import models

from utils.torch.trainer import TorchTrainer


class FasterRCNNTrainer(TorchTrainer):
    def __init__(
        self,
        network: Union[models.detection.FasterRCNN, models.detection.FasterRCNN],
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        images, targets = batch

        # Move inputs to cpu or gpu
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.clone().detach().to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.network(images, targets)  # the model computes the loss automatically if we pass in targets
        loss = sum(loss for loss in loss_dict.values())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.detach().cpu().item()}

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # We not use eval() because the loss is not calculated in the eval mode
        self.network.train()
        # Prepare batch
        images, targets = batch

        # Move inputs to cpu or gpu
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.clone().detach().to(self.device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            # Forward pass
            loss_dict = self.network(images, targets)  # the model computes the loss automatically if we pass in targets
            loss = sum(loss for loss in loss_dict.values())

        return {"loss": loss.detach().cpu().item()}
