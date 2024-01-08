import torch
import torch.nn as nn
import numpy as np

class Gaussian_Noise(nn.Module):
    def __init__(self, mean=0, std=0.06):
        super(Gaussian_Noise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, noised):
        noise = torch.normal(torch.full_like(noised, self.mean), torch.full_like(noised, self.std))
        noised_image = noised + noise
        return noised_image


class GN(nn.Module):

    def __init__(self, var=0.06, mean=0):
        super(GN, self).__init__()
        self.var = var
        self.mean = mean

    def gaussian_noise(self, image, mean, var):
        noise = torch.Tensor(np.random.normal(mean, var ** 0.5, image.shape)).to(image.device)
        out = image + noise
        return out

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.gaussian_noise(image, self.mean, self.var)
