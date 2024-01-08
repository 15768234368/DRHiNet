import torch
import torch.nn as nn

'''
椒盐噪声椒盐噪声的概率参数通常取值范围在 [0, 1] 之间，其中 0 表示没有椒盐噪声，1 表示每个像素都变为椒盐噪声。然而，选择合适的概率参数并没有一个固定的标准，而是要根据实际图像的情况和应用需求来决定。

如果你希望模拟轻微的噪声，可以选择一个较小的概率值，如 0.01 或更小。这将在图像中引入少量的椒盐噪声，可以帮助测试算法对噪声的鲁棒性。

如果你想要更明显的噪声效果，可以选择较大的概率值，如 0.1 或更大。这将在图像中引入较多的椒盐噪声，可以用来测试算法对噪声的处理能力。

'''
class SP(nn.Module):

    def __init__(self, prob=0.05):
        super(SP, self).__init__()
        self.prob = prob

    def sp_noise(self, image, prob):
        prob_zero = prob / 2
        prob_one = 1 - prob_zero
        rdn = torch.rand(image.shape).to(image.device)

        output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
        output = torch.where(rdn < prob_zero, torch.ones_like(image).to(image.device), output)

        return output

    def forward(self, image):
        noise_image = self.sp_noise(image, self.prob)
        return noise_image