import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
import os
from noise_layers.gaussian_noise import Gaussian_Noise
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""高斯噪声"""
gaussian = Gaussian_Noise().to(device)

def load(name):
    state_dicts = torch.load(name, map_location=torch.device(device))
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
init_model(net)
net.to(device)

params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()


with torch.no_grad():
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        cover = data[:]
        secret = torch.Tensor(np.random.choice([0, 1], size=[cover.shape[0], 1, c.messages_length, c.messages_length]))
        secret = secret.expand(-1, 3, -1, -1).to(device)
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
        backward_z = gauss_noise(output_z.shape)

    
        #################
        #    add noise  #
        #################        
        noise_image = gaussian(steg_img.to(device))
        output_steg = dwt(noise_image)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((output_steg, backward_z), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
        cover_rev = iwt(cover_rev)
        resi_cover = (steg_img - cover) * 20
        resi_secret = (secret_rev - secret) * 20
        # 如果目录不存在，则创建它
        if not os.path.exists(c.IMAGE_PATH_cover):
            os.makedirs(c.IMAGE_PATH_cover)
        if not os.path.exists(c.IMAGE_PATH_secret):
            os.makedirs(c.IMAGE_PATH_secret)
        if not os.path.exists(c.IMAGE_PATH_steg):
            os.makedirs(c.IMAGE_PATH_steg) 
        if not os.path.exists(c.IMAGE_PATH_secret_rev):
            os.makedirs(c.IMAGE_PATH_secret_rev)         
        if not os.path.exists(c.IMAGE_PATH_attack):
            os.makedirs(c.IMAGE_PATH_attack)                
        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
        torchvision.utils.save_image(noise_image, c.IMAGE_PATH_attack + '%.5d.png' % i)
        decoded_rounded = torch.round(secret_rev.detach()).clamp(0, 1)
        bitwise_err = torch.sum(torch.abs(decoded_rounded - secret.detach())) / torch.tensor((secret.shape[0] * secret.shape[1] * secret.shape[2] * secret.shape[3]), device=device)

        print(f"比特错误率：{bitwise_err}")




