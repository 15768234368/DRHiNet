#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import attack
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.cuda()


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.cuda()


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.cuda()


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def makeDirs():
    file_dir = "runs-works"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # 获取当前日期和时间
    current_time = datetime.datetime.now()
    # 格式化日期和时间
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # name = "run" + "_" + "base" + "_" + time_str
    # name = "run" + "_" + "Gaussian-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Quantization-no-normalize" + "_" + time_str
    name = "run" + "_" + "Jpeg-50-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Brightness-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Contrast(0.5-0.6)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Saturation(0.5-0.6)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Hue(0.2-0.3)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Brightness(1.5-1.6)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Contrast(1.5-1.6)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Saturation(1.5-1.6)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "Hue(（-0.3）-（-0.2）)-no-normalize" + "_" + time_str
    # name = "run" + "_" + "salt-no-normalize" + "_" + time_str

    train_dir = os.path.join(file_dir, name)
    os.makedirs(train_dir, exist_ok=True)
    train_save_model = os.path.join(train_dir, 'model')
    os.makedirs(train_save_model, exist_ok=True)
    return train_save_model


train_save_model = makeDirs()
#run_name = "Contrast(1.5-1.6)-no-normalize"
# run_name = "Gaussian-no-normalize"
# run_name = "Hue(（-0.3）-（-0.2）)-no-normalize"
run_name = "Jpeg-50-no-normalize"
#####################
# Model initialize: #
#####################
net = Model()
init_model(net)
net.cuda()


attack_model = attack.Attack()
attack_model = attack_model.cuda()

para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)

try:
    writer = SummaryWriter(comment=run_name, filename_suffix="steg")

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []

        #################
        #     train:    #
        #################

        for i_batch, data in enumerate(datasets.trainloader):
            data = data.cuda()
            cover = data[:]
            secret = torch.Tensor(np.random.choice([0, 1], size=[cover.shape[0], 1, c.messages_length, c.messages_length]))
            secret = secret.expand(-1, 3, -1, -1).cuda()
            
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

            #################
            #    add noise  #
            #################
            if c.attack:
                # attack_state_dict = torch.load('attack_runs/run-Gaussian(0.06)_2023-11-18_22-53-29/model/best_model.pth')
                # attack_state_dict = torch.load('attack_runs/run-Brightness_2023-11-18_22-23-30/model/best_model.pth', map_location=torch.device(device))
                # attack_state_dict = torch.load('attack_runs/run-Contrast_2023-11-18_22-27-20/model/best_model.pth', map_location=torch.device(device))
                # attack_state_dict = torch.load('attack_runs/run-Gaussian(0.06)_2023-11-18_22-53-29/model/best_model.pth', map_location=torch.device(device))
                # attack_state_dict = torch.load('attack_runs/run-Hue_2023-11-18_22-48-15/model/best_model.pth', map_location=torch.device(device))
                attack_state_dict = torch.load('attack_runs/run-JPEG(50)_2023-11-18_22-49-09/model/best_model.pth', map_location=torch.device(device))
                attack_model.load_state_dict(attack_state_dict)
                output_steg_attack = attack_model(steg_img.cuda())
                output_steg_attack = dwt(output_steg_attack)

            #################
            #   backward:   #
            #################

            output_z_guass = gauss_noise(output_z.shape)
            output_rev = torch.cat((output_steg_attack, output_z_guass), 1)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev).cuda()

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #     val:    #
        #################
        bitwise_err = []
        psnr_c = []
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                net.eval()
                for x in datasets.testloader:
                    x = x.cuda()
                    cover = x[:]
                    secret = torch.Tensor(np.random.choice([0, 1], size=[cover.shape[0], 1, c.messages_length, c.messages_length])).cuda()
                    secret = secret.expand(-1, 3, -1, -1).cuda()
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #    add noise  #
                    #################
                    if c.attack:
                        output_steg_attack = attack_model(steg.cuda())
                        output_steg_attack = dwt(output_steg_attack)    

                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg_attack, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)
                    decoded_rounded = torch.round(secret_rev.detach()).clamp(0, 1)
                    bitwise_err_temp = torch.sum(torch.abs(decoded_rounded - secret.detach())) / torch.tensor((secret.shape[0] * secret.shape[1] * secret.shape[2] * secret.shape[3]), device=device)
                    bitwise_err.append((bitwise_err_temp).cpu().numpy())

                writer.add_scalars("Bitwise_err", {"average bitwise_err": np.mean(bitwise_err)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
        viz.show_loss(epoch_losses, run_name, np.mean(bitwise_err), np.mean(psnr_c))
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, train_save_model + '/model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()},  + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, train_save_model + '/model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
