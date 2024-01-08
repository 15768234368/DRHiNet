import torch.nn as nn


class Attack(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Attack, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image):
        attack_image = self.conv_layer(image) + image
        return attack_image