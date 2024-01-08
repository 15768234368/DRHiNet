import torch.nn as nn
from kornia.filters import GaussianBlur2d

"""
	高斯滤波器
	sigma: 取值0.5~5，越小，细节保留越好，越大，图像越光滑，小一般为1
"""


class GF(nn.Module):

    def __init__(self, sigma=0.7, kernel=7):
        super(GF, self).__init__()
        self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

    def forward(self, image):
        images = image
        return self.gaussian_filter(images)