import copy
import glob
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision import datasets

from configs.base import Config


def get_transforms(opt: Config, train=False):
    if train:
        return A.Compose(
            [
                A.Resize(opt.input_size_width, opt.input_size_height),
                A.HorizontalFlip(p=opt.horizontal_flip),
                A.VerticalFlip(p=opt.vertical_flip),
                A.RandomBrightnessContrast(p=opt.brightness_contrast),
                A.ColorJitter(p=opt.color_jitter),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco"),
        )
    return A.Compose(
        [
            A.Resize(opt.input_size_width, opt.input_size_height),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco"),
    )


class CocoDataset(datasets.VisionDataset):
    def __init__(self, root, split="train", **kwargs):
        super().__init__(root, **kwargs)
        self.split = split  # train, valid, test
        self.coco = COCO(os.path.join(root, "annotations", f"instances_{split}.json"))  # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resize_and_pad(self, image, target):
        pass

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t["bbox"] + [t["category_id"]] for t in target]  # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"] for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"] for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # we have a different area
        targ["iscrowd"] = torch.tensor([t["iscrowd"] for t in target], dtype=torch.int64)
        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.ids)


class YoLoDataset(datasets.VisionDataset):
    def __init__(self, root, split="train", **kwargs):
        super().__init__(root, **kwargs)
        self.split = split  # train, valid, test
        img_paths = glob.glob(os.path.join(root, split, "images", "*"))
        label_paths = []
        for path in img_paths:
            img_path, _ = os.path.splitext(path)
            label_path = img_path.replace("images", "labels") + ".txt"
            label_paths.append(label_path)

        self.ids = list(zip(img_paths, label_paths))

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resize_and_pad(self, image, target):
        pass

    def _load_target(self, path: str) -> List:
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            lines = [[float(x) for x in line] for line in lines]
        return lines

    def __getitem__(self, index):
        id = self.ids[index]
        image_id = int(os.path.basename(id[0].split(".")[0]))
        image = self._load_image(id[0])
        target = self._load_target(id[1])

        bboxes = []
        cls_ids = []
        for t in target:
            cls_id = t[0]
            # rescale to 0 - image_size
            box = np.array(t[1:]) * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            box = list(box)
            # convert from xcycwh to xywh
            box[0] = box[0] - box[2] / 2
            box[1] = box[1] - box[3] / 2

            bboxes.append(box + [cls_id])
            cls_ids.append(cls_id)

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([x for x in cls_ids], dtype=torch.int64)
        targ["image_id"] = torch.tensor([image_id for _ in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # we have a different area
        targ["iscrowd"] = torch.tensor([0 for t in target], dtype=torch.int64)
        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.ids)


def build_train_test_dataset(opt: Config, val_split: str = "val") -> Tuple[DataLoader, DataLoader]:
    """Build train and test dataset

    Args:
        opt (Config): Config object
        val_split (str, optional): validation split name, change this to "test" if you want to test on test set. Defaults to "val".

    Returns:
        Tuple[DataLoader, DataLoader]: train and test dataloader
    """
    dataset = {
        "CocoDataset": CocoDataset,
        "YoLoDataset": YoLoDataset,
    }

    def _collate_fn(batch):
        return tuple(zip(*batch))

    dataset_fn = dataset.get(opt.data_format, None)

    assert dataset_fn is not None, f"Dataset {opt.data_format} is not implemented"

    train_dataset = dataset_fn(root=opt.data_root, split="train", transforms=get_transforms(opt, True))
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=_collate_fn)

    test_dataset = dataset_fn(root=opt.data_root, split=val_split, transforms=get_transforms(opt, False))
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=_collate_fn)

    return (train_dataloader, test_dataloader)
