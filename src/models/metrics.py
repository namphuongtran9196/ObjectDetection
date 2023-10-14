import json
import os
from typing import Dict, List

import numpy as np
from mean_average_precision import MetricBuilder
from mean_average_precision.metric_base import MetricBase
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


##################################################################################################
######################################## COCO mAP Evaluator ######################################
##################################################################################################
class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JsonNumpyEncoder, self).default(obj)


def coco_mAP(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    classes_ids: List[int] = None,
    summarize: bool = True,
) -> Dict[str, float]:
    """Calculate the mean average precision (mAP) for the COCO dataset.

    Args:
        ground_truth (np.ndarray): The ground truth bounding boxes and classes.
                    Must be in the format: [[xmin, ymin, xmax, ymax, class_id], ...] and 3 dimensional.
                                            1st dimension is the number of images.
                                            2nd dimension is the number of bounding boxes in the image.
                                            3rd dimension is the bounding box coordinates and class id.
                    Sample ground truth:
                        gt = np.array([
                            [ # image 1
                                [10, 10, 50, 50, 0], # bounding box gt 1
                                [20, 20, 60, 60, 1], # bounding box gt 2
                            ],
                            [ # image 2
                                [20, 20, 60, 60, 1], # bounding box gt 1
                            ]
                        ]

        predictions (np.ndarray): The predicted bounding boxes and classes.
                    Must be in the format: [[xmin, ymin, xmax, ymax, class_id, confidence], ...] and 3 dimensional.
                                            1st dimension is the number of images.
                                            2nd dimension is the number of bounding boxes in the image.
                                            3rd dimension is the bounding box coordinates, class id and confidence.
                    Sample predictions:
                        preds = np.array([
                            [ # image 1
                                [10, 10, 50, 50, 0, 0.9], # bounding box prediction 1
                                [20, 20, 60, 60, 1, 0.8], # bounding box prediction 2
                            ],
                            [ # image 2
                                [20, 20, 60, 60, 1, 0.7], # bounding box prediction 1
                                [20, 20, 60, 60, 1, 0.1], # bounding box prediction 2
                            ]
                        ]
        Note: the order of images must be the same for ground truth and predictions.

        num_classes (int): The number of classes in the dataset.
        classes_ids (List[int]): Use for calculating mAP for a subset of classes.
                                For example:
                                    Full dataset classes_ids = [0,1,2,3,4,5,6] or classes_ids = None
                                    Calculate mAP for 3 only: classes_name = [3]
        summarize (bool): Whether to print the mAP results to the console.
    Returns:
        Dict[str, float]: The mean average precision (mAP.5:.95 and mAP.5) for the COCO dataset.
                        Dict keys: "mAP.5:.95" and "mAP.5"
    """
    # Create ground truth data in COCO format
    gt_data = {"images": [], "annotations": [], "categories": []}

    # Add category information
    for class_label in range(num_classes):
        category_info = {"id": class_label, "name": class_label, "supercategory": "object"}
        gt_data["categories"].append(category_info)

    # Add image and annotation information
    annotation_id = 1  # Assign a unique ID to each annotation
    for image_id, image_annotations in enumerate(ground_truth):
        image_info = {"id": image_id}
        gt_data["images"].append(image_info)

        for annotation in image_annotations:
            xmin, ymin, xmax, ymax, class_label = annotation

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_label,
                # Convert from xmin, ymin, xmax, ymax to xmin, ymin, width, height
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,  # Assuming no crowd annotations
            }
            gt_data["annotations"].append(annotation_info)

            annotation_id += 1

    # Create a COCO dataset object
    gt_coco = COCO()
    # Set the dataset's attributes using your JSON data
    gt_coco.dataset = gt_data
    gt_coco.createIndex()

    # Create prediction data in COCO format
    pred_results = []
    for image_id, image_preds in enumerate(predictions):
        for pred in image_preds:
            xmin, ymin, xmax, ymax = pred[:4]

            result = {
                "image_id": image_id,  # ID of the corresponding image
                "category_id": int(pred[4]),  # ID of the predicted class
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # Bounding box coordinates [top, left, width, height]
                "score": float(pred[5]),  # Confidence score of the prediction
            }
            pred_results.append(result)
    pred_coco = gt_coco.loadRes(pred_results)

    # Evaluate the COCO metric for bounding box detection
    coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
    if classes_ids is not None:
        coco_eval.params.catIds = classes_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    if summarize:
        coco_eval.summarize()

    return {"mAP.5:.95": coco_eval.stats[0], "mAP.5": coco_eval.stats[1]}


##################################################################################################
######################################## VOC mAP Evaluator #######################################
##################################################################################################

"""
A modified version of the original code from https://github.com/Cartucho/mAP
"""
