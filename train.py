import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import datetime
import os
import sys
from utils import *
from network import UNet
CE_loss = nn.CrossEntropyLoss()

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use")

parser.add_argument('--batch_size', type=int, default=16, help="the batch_size")

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")

parser.add_argument("--sample_mode", type=int, default=1, help="Down-sampling mode: 1-interval, 2-continuous, 3-random")

parser.add_argument('--data_dir', type=str, default='dataset/Cy5_Data0612_vi/',
                    help="A folder containing files for training")
parser.add_argument('--data_name', type=str, default='train_16_25', help="A folder containing files for training")
parser.add_argument('--pth_path', type=str, default='checkpoints', help="pth file root path")

parser.add_argument('--interval', type=int, default=20, help="save pth file per {} epochs")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
# tensorboard --logdir="D:\BBBB_Segment\Spec2Spec_Code\pth_number\Cy5_Data0612_vi_train_16_25_sample1_202401161004\logs" --host=127.0.0.1


def train_epoch():
    network.train()

    total_loss_list = 0
    for iteration, (noisy) in enumerate(trainloader):

        noisy = noisy.cuda()
        if opt.sample_mode == 1:
            # 间隔采样（连续）
            spe_1, spe_2, spe_3 = generate_subimages(noisy)
        elif opt.sample_mode == 2:
            # 连续采样
            spe_1, spe_2, spe_3 = generate_subimages_2(noisy)
        elif opt.sample_mode == 3:
            # 间隔采样（随机）
            spe_1, spe_2, spe_3 = generate_subimages_3(noisy)
        else:
            spe_1, spe_2, spe_3 = generate_subimages_4(noisy)

        noisy_output = network(spe_2)
        loss2neighbor_1 = 0.5 * L1_pixelwise(noisy_output, spe_1) + 0.5 * L2_pixelwise(noisy_output, spe_1)
        loss2neighbor_2 = 0.5 * L1_pixelwise(noisy_output, spe_3) + 0.5 * L2_pixelwise(noisy_output, spe_3)

        ################################################################################################################
        # Total loss
        Total_loss = 0.5 * loss2neighbor_1 + 0.5 * loss2neighbor_2
        Total_loss.backward()
        optimizer.step()

        ################################################################################################################
        total_loss_list += Total_loss.item()

        elapsed_time = datetime.datetime.now() - start_time
        if iteration % 1 == 0:
            print(
                '\r[Epoch %d/%d]  [Batch %d/%d]  [Total loss: %.2f]  [Time Cost: %s]'
                % (
                    epoch + 1,
                    opt.n_epochs,
                    iteration + 1,
                    len(trainloader),
                    total_loss_list/train_number,
                    elapsed_time
                ), end=' ')

        if (iteration + 1) % len(trainloader) == 0:
            print('\n', end=' ')

    return total_loss_list/train_number


if __name__ == '__main__':

    ##################################################### create file dir ##############################################
    root_dir = opt.data_dir.split('/')[-2]
    current_time = opt.data_dir.split('/')[-2] + "_" + opt.data_name + \
                   "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

    pth_root_path = opt.pth_path + '/' + current_time
    pth_path = pth_root_path + "/model"
    log_path_name = pth_root_path + '/logs'

    print("ckp is saved in {}".format(pth_root_path))
    if not os.path.exists(pth_root_path):
        os.mkdir(pth_root_path)
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    if not os.path.exists(log_path_name):
        os.mkdir(log_path_name)


    ################################################## load dataset ####################################################
    path = opt.data_dir + "/" + opt.data_name
    train_data = Images_Dataset(data_dir=path)

    train_number = len(train_data)
    print("train_70 dataset: " + str(train_number))
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    network = UNet(in_channels=1, out_channels=1, bilinear=False)
    if torch.cuda.is_available():
        network = network.cuda()
        L2_pixelwise.cuda()
        L1_pixelwise.cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ################################################ training process ##################################################
    start_time = datetime.datetime.now()

    min_train_loss = sys.maxsize
    min_train_loss_epoch = 0

    for epoch in range(0, opt.n_epochs):
        train_loss = train_epoch()

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = epoch + 1
            model_save_name = pth_path + '//E_Min_Train_Loss.pth'
            torch.save(network.state_dict(), model_save_name)

        # save model
        if (epoch + 1) % opt.interval == 0:
            model_save_name = pth_path + '//E_' + str(epoch + 1).zfill(2) + '.pth'
            torch.save(network.state_dict(), model_save_name)

    elapsed_time = datetime.datetime.now() - start_time
    print('Training process has completed!   [Time Cost: %s]' % elapsed_time)
    print('[Epoch: %d]   [Min Training Loss: %.2f]' % (min_train_loss_epoch, min_train_loss))


















