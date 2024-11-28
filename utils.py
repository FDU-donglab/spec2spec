import os
import glob
import torch
import numpy as np
import random
import tifffile as tiff
from torch.utils.data import Dataset
import scipy.io as scio


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


class Images_Dataset(Dataset):
    """
        read tiff images
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.paths = glob.glob(os.path.join(self.data_dir, "*"))

        self.images = tiff.imread(self.paths[0]).astype('float32')
        print("Load data: " + self.paths[0])
        for i in range(1, len(self.paths)):
            images = tiff.imread(self.paths[i]).astype('float32')
            self.images = np.concatenate([images, self.images], axis=0)
            print("Load data: " + self.paths[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        noisy = self.images[idx]

        noisy = torch.from_numpy(noisy).unsqueeze(dim=0)
        return noisy


class Add_Noise_Images_Dataset(Dataset):
    """
        read tiff images
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.paths = glob.glob(os.path.join(self.data_dir, "*"))

        self.images = tiff.imread(self.paths[0]).astype('float32')
        for i in range(1, len(self.paths)):
            images = tiff.imread(self.paths[i]).astype('float32')
            self.images = np.concatenate([images, self.images], axis=0)

    def __len__(self):
        return len(self.images)

    def __add_noise(self, clean):
        # 泊松噪声
        vals = len(np.unique(clean))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_1 = np.random.poisson(clean * vals) / float(vals)

        min_std, max_std = 0, 0.2
        std = np.random.rand(1, 1) * (max_std - min_std) + min_std
        guass = np.random.normal(
            loc=0.0,
            scale=std,
            size=noisy_1.shape
        )

        noisy = prctile_norm(noisy_1) + prctile_norm(guass)
        noisy = prctile_norm(noisy) * 1000
        return noisy

    def __getitem__(self, idx):
        clean = self.images[idx]
        noisy = self.__add_noise(clean)
        noisy = noisy.astype('float32')

        # tiff.imwrite("Noisy_1.tif", noisy.astype(np.uint16))

        noisy = torch.from_numpy(noisy).unsqueeze(dim=0)
        clean = torch.from_numpy(clean).unsqueeze(dim=0)
        return noisy, clean


class Classify_Dataset(Dataset):
    """
        read tiff images and labels
    """

    def __init__(self, data_dir, img_dir='images', label_dir='labels'):
        self.image_dir = os.path.join(data_dir, img_dir)
        self.label_dir = os.path.join(data_dir, label_dir)

        self.img_paths = glob.glob(os.path.join(self.image_dir, "*"))
        self.label_paths = glob.glob(os.path.join(self.label_dir, "*"))

        self.images = tiff.imread(self.img_paths[0]).astype('float32')
        self.labels = scio.loadmat(self.label_paths[0])['label']

        for i in range(1, len(self.img_paths)):
            images = tiff.imread(self.img_paths[i]).astype('float32')
            self.images = np.concatenate([images, self.images], axis=0)

            labels = scio.loadmat(self.label_paths[i])['label']
            self.labels = np.concatenate([labels, self.labels], axis=1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        noisy = self.images[idx]
        noisy = torch.from_numpy(noisy).unsqueeze(dim=0)

        # label = self.labels[0, idx]
        # print(idx)
        label = torch.zeros(2)
        label[self.labels[0, idx]] = 1

        # print()
        return noisy / 65535.0, label


class Input_Mat_Dataset(Dataset):
    """
        read mat. file
    """

    def __init__(self, data_dir, img_dir, label_dir):
        self.data_dir = data_dir
        self.img_path = glob.glob(os.path.join(self.data_dir, img_dir, "*"))
        self.label_path = glob.glob(os.path.join(self.data_dir, label_dir, "*"))

        self.images = scio.loadmat(self.img_path[0])['images']
        self.labels = scio.loadmat(self.label_path[0])['label']

        self.global_min_value = np.min(self.images)
        self.images = self.images.astype('float32')
        self.size = self.images.shape[2]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        noisy = self.images[:, :, idx] + self.global_min_value
        noisy = torch.from_numpy(noisy).unsqueeze(dim=0)

        # label = np.zeros(2).astype('int64')
        # if self.labels[:, idx] == 0:
        #     label[1] = 1
        # else:
        #     label[0] = 1
        # label = torch.from_numpy(label)

        return noisy


# 间隔采样
def generate_subimages(img):
    """
        fixed downsampling
    """
    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(w // 3):
        sub_1[:, :, :, i] = img[:, :, :, i * 3]
        sub_2[:, :, :, i] = img[:, :, :, i * 3 + 1]
        sub_3[:, :, :, i] = img[:, :, :, i * 3 + 2]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)

    return spe_1, spe_2, spe_3


# 连续采样
def generate_subimages_2(img):
    """
        fixed downsampling
    """
    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)

    sub_1[:, :, :, :] = img[:, :, :, 0: 5]
    sub_2[:, :, :, :] = img[:, :, :, 5: 10]
    sub_3[:, :, :, :] = img[:, :, :, 10: 15]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)

    return spe_1, spe_2, spe_3

# 随机采样
def generate_subimages_3(img):
    """
        random downsampling
    """
    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(w // 3):
        idx = [0, 1, 2]
        random.shuffle(idx)

        sub_1[:, :, :, i] = img[:, :, :, i * 3 + idx[0]]
        sub_2[:, :, :, i] = img[:, :, :, i * 3 + idx[1]]
        sub_3[:, :, :, i] = img[:, :, :, i * 3 + idx[2]]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)

    return spe_1, spe_2, spe_3


def generate_subimages_4(img):
    """
        random downsampling
    """
    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 3, dtype=img.dtype, layout=img.layout, device=img.device)

    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    random.shuffle(idx)
    for i in range(w // 3):
        sub_1[:, :, :, i] = img[:, :, :, idx[i]]

    for i in range(w // 3):
        sub_2[:, :, :, i] = img[:, :, :, idx[i + 5]]

    for i in range(w // 3):
        sub_3[:, :, :, i] = img[:, :, :, idx[i + 10]]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)

    return spe_1, spe_2, spe_3


def generate_two_samplings(img):
    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(w // 3):
        sub_1[:, :, :, i] = img[:, :, :, i * 2]
        sub_2[:, :, :, i] = img[:, :, :, i * 2 + 1]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    return spe_1, spe_2


def generate_four_samplings(img):
    n, c, h, w = img.shape
    img_padding = torch.zeros(n, c, h, w + 1, dtype=img.dtype, layout=img.layout, device=img.device)
    img_padding[:, :, :, 0:w] = img
    img_padding[:, :, :, w] = img[:, :, :, 0]

    n, c, h, w = img_padding.shape
    sub_1 = torch.zeros(n, c, h, w // 4, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 4, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 4, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_4 = torch.zeros(n, c, h, w // 4, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(w // 4):
        sub_1[:, :, :, i] = img_padding[:, :, :, i * 4]
        sub_2[:, :, :, i] = img_padding[:, :, :, i * 4 + 1]
        sub_3[:, :, :, i] = img_padding[:, :, :, i * 4 + 2]
        sub_4[:, :, :, i] = img_padding[:, :, :, i * 4 + 3]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)
    spe_4 = torch.sum(sub_4, dim=3)

    return spe_1, spe_2, spe_3, spe_4

def generate_five_samplings(img):

    n, c, h, w = img.shape
    sub_1 = torch.zeros(n, c, h, w // 5, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_2 = torch.zeros(n, c, h, w // 5, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_3 = torch.zeros(n, c, h, w // 5, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_4 = torch.zeros(n, c, h, w // 5, dtype=img.dtype, layout=img.layout, device=img.device)
    sub_5 = torch.zeros(n, c, h, w // 5, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(w // 4):
        sub_1[:, :, :, i] = img[:, :, :, i * 5]
        sub_2[:, :, :, i] = img[:, :, :, i * 5 + 1]
        sub_3[:, :, :, i] = img[:, :, :, i * 5 + 2]
        sub_4[:, :, :, i] = img[:, :, :, i * 5 + 3]
        sub_5[:, :, :, i] = img[:, :, :, i * 5 + 4]

    spe_1 = torch.sum(sub_1, dim=3)
    spe_2 = torch.sum(sub_2, dim=3)
    spe_3 = torch.sum(sub_3, dim=3)
    spe_4 = torch.sum(sub_4, dim=3)
    spe_5 = torch.sum(sub_5, dim=3)

    return spe_1, spe_2, spe_3, spe_4, spe_5



if __name__ == "__main__":
    noisy = torch.randn((2, 1, 42, 15)).cuda()
    # sub_1, sub_2, sub_3 = generate_subimages_4(img=noisy)
    # sub_1, sub_2 = generate_two_samplings(img=noisy)
    sub_1, sub_2, sub_3, sub_4 = generate_four_samplings(img=noisy)

    print(noisy.shape)
    print(sub_1.shape)
    print(sub_2.shape)
    print(sub_3.shape)
    print(sub_4.shape)
