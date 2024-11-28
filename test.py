import os
import torch
import argparse
import time
import datetime
import numpy as np
from utils import Images_Dataset
from matplotlib import pyplot as plt
from network import UNet
from torch.utils.data import DataLoader
from scipy.io import savemat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation")
parser.add_argument('--batch_size', type=int, default=1, help="batch_size")

parser.add_argument('--image_dir', type=str, default='dataset/Cy5_Data0612_vi/data_fig_4_3/test_batch/',
                    help="A folder containing files to be tested")
# model parameter
parser.add_argument('--pth_path', type=str, default='checkpoints', help="pth file root path")
parser.add_argument('--denoise_model', type=str, default='Cy5_Data0612_vi_train_16_25_sample1_202401161004/model',
                    help='A folder containing models to be tested')
parser.add_argument('--pth_index', type=str, default='200', help='A folder containing models to be tested')


parser.add_argument('--result_save_dir', type=str, default='dataset/Cy5_Data0612_vi/data_fig_4_3/result/',
                    help='A folder to save predicted result')

parser.add_argument('--is_save_result', type=bool, default=True, help='if save predicted result')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)

# pth_200_test_batch_TData0612_Cy5_-16_25_10000.mat
if __name__ == "__main__":
    suffix = "TData0612_Cy5_-16_25_10000"
    filename = opt.result_save_dir + "/pth_" + opt.pth_index + "_" + opt.image_dir.split('/')[-2] + '_' + suffix + '.mat'

    network = UNet(in_channels=1, out_channels=1, bilinear=False).cuda()

    # load model
    model_name = opt.pth_path + '//' + opt.denoise_model + "//E_" + opt.pth_index + ".pth"
    network.load_state_dict(torch.load(model_name))

    test_data = Images_Dataset(data_dir=opt.image_dir)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    raw = []
    predict = []

    start_time = datetime.datetime.now()
    with torch.no_grad():
        for iteration, noisy in enumerate(test_loader):
            noisy = noisy.cuda()
            noisy = torch.sum(noisy, dim=3)

            fake_B = network(noisy)

            output = np.squeeze(fake_B.cpu().detach().numpy())
            noisy = np.squeeze(noisy.cpu().detach().numpy())

            raw.append(noisy)
            predict.append(output)
            if (iteration + 1) % 100 == 0:
                print('\r[Batch %d/%d]' % (iteration + 1, len(test_loader)), end=' ')

    time_end = time.time()
    elapsed_time = datetime.datetime.now() - start_time
    print('\nPredicted process has completed!   [Time Cost: %s]' % elapsed_time)

    for i in range(50, 55):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(raw[i])
        plt.subplot(1, 2, 2)
        plt.plot(predict[i])
        # plt.show()

    # -------------------------------------------save data ------------------------------------------------
    raw_data = PCA(n_components=2).fit_transform(raw)
    # raw_labels = mixture.GaussianMixture(n_components=2).fit(raw_data).predict(raw_data)
    raw_labels = KMeans(n_clusters=2, random_state=123).fit(raw_data).labels_
    # raw_labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(raw_data)

    output_data = PCA(n_components=2).fit_transform(predict)
    # predict_labels = mixture.GaussianMixture(n_components=2).fit(output_data).predict(output_data)
    predict_labels = KMeans(n_clusters=2, random_state=123).fit(output_data).labels_
    # predict_labels = DBSCAN(eps=4, min_samples=5).fit_predict(output_data)


    print(predict_labels)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(raw_data[:, 0], raw_data[:, 1], s=1, c=raw_labels)
    plt.subplot(1, 2, 2)
    plt.scatter(output_data[:, 0], output_data[:, 1], s=1, c=predict_labels)
    # plt.show()

    if opt.is_save_result:
        print("Save Result Successfully !")
        savemat(filename, {'predict': predict,
                           'raw': raw,
                           'predict_labels': predict_labels,
                           'raw_labels': raw_labels,
                           'PCA_raw_data': raw_data,
                           'PCA_denoised_data': output_data
                           })



