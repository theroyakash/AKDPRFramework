"""
Implementation of YOLO Loss algorithm from original YOLO paper

Usage: from AKDPRFramework.applications.yololoss import YoloLoss
"""

import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, split_size, nb_boxes, nb_classes):
        """
        YoloLoss initializer
            Args:
                split_size: split_size for the image. In original paper split_size was 7
                nb_boxes: number of boxes original paper has implemented 2
                nb_classes: number of classes. Pascal VOC dataset which was used by the paper was 20
        """
        super(YoloLoss, self).__init__()
        self.split_size = split_size
        self.nb_boxes = nb_boxes
        self.nb_classes = nb_classes

        # Mentioned in the paper the following variables indicates how much we should pay loss for no object and box
        # coordinates
        self.lambda_no_object = 0.5
        self.lambda_no_coordinates = 5

        