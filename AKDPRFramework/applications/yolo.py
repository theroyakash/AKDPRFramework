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
