import torch.nn as nn
from kornia.filters import MedianBlur

"""
	中值滤波器，去除椒盐噪声
	
	一般来说，kernel 的取值应该根据图像中噪声的情况进行选择：

    对于较小的噪声点，kernel 可以取较小的值，如 3x3 或 5x5。这样的窗口足够小，能够捕获到小范围内的噪声。

    对于较大的噪声区域，或者在噪声较为严重的情况下，可以考虑使用较大的 kernel 值，如 7x7 或更大。这样的窗口能够覆盖更大范围的像素，从而更好地去除噪声。

    需要注意的是，kernel 越大，计算中值的成本就越高，因为要考虑更多的像素。因此，选择适当的 kernel 大小要权衡计算成本和去噪效果。
"""

class MF(nn.Module):

    def __init__(self, kernel=3):
        super(MF, self).__init__()
        self.middle_filter = MedianBlur((kernel, kernel))

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.middle_filter(image)
