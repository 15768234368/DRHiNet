import torch.nn as nn
import torchvision.transforms as transforms

class ColorJitter(nn.Module):
    """
        Brightness:亮度   2
        Contrast:对比度    2
        Saturation:饱和度  2
        Hue:色相  0.1
    """
    def __init__(self, distortion):
        super(ColorJitter, self).__init__()
        #
        brightness = [1.5, 1.6]
        contrast = [1.5, 1.6]
        saturation = [1.5, 1.6]
        hue = [-0.3, -0.2]
        #
        if distortion == 'Brightness':
            self.transform = transforms.ColorJitter(brightness=brightness)
        if distortion == 'Contrast':
            self.transform = transforms.ColorJitter(contrast=contrast)
        if distortion == 'Saturation':
            self.transform = transforms.ColorJitter(saturation=saturation)
        if distortion == 'Hue':
            self.transform = transforms.ColorJitter(hue=hue)

    def forward(self, image):
        # #
        # noise_img = (image + 1) / 2   # [-1, 1] -> [0, 1]
        # #
        # ColorJitter = self.transform(watermarked_img)
        # #
        # ColorJitter = (ColorJitter * 2) - 1  # [0, 1] -> [-1, 1]
        ColorJitter = self.transform(image)

        return ColorJitter


