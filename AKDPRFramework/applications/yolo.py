"""
PyTorch Implementation of Deep Learning Framework YOLO (Original paper)
With slight modifications of added Dropout and BatchNormalization

This module contains the Model Architecture for YOLO Model is a part of AKDPRFramework
Author: ©2020 theroyakash
"""

import torch
import torch.nn as nn

architecture_config = [
#  (kernel_size, filter amt, strides, padding)
    (7, 64, 2, 3),
    "Maxpooling",
    (3, 192, 1, 1),
    "Maxpooling",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "Maxpooling",
#   (kernel_size, filter amt, strides, padding), sequential repeatation time
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "Maxpooling",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyReLU(self.batchnorm(self.conv(x)))


class YOLOOriginal(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOOriginal, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                number_of_repeats = x[2]

                for _ in range(number_of_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, nb_boxes, nb_classes):
        """
        Creates the fully connected layer at the end.
            Args:
                - split_size: (int) Mention the split size
                - nb_boxes: Number of BOXES
                - nb_classes: number of classes

            Returns:
                Sequential pytorch model.
        """

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 496),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (nb_classes + nb_boxes * 5))
        )
