import glob
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)


class MemoryFriendlyLoader_zy(torch.utils.data.Dataset):
    def __init__(self, low_img_dir, high_img_dir, task, batch_w, batch_h, nbins=14, exp_mean=0.55, is_train=True):
        self.exp_mean = exp_mean
        self.task = task
        self.train_low_data_names = []
        self.train_high_data_names = []
        self.nbins = nbins
        self.is_train = is_train

        self.batch_w = batch_w
        self.batch_h = batch_h

        self.train_low_data_names = glob.glob(low_img_dir + "*.*")
        self.train_high_data_names = glob.glob(high_img_dir + "*.*")

        self.count = len(self.train_low_data_names)
        self.low_data = []
        self.high_data = []
        self.hist_data = []

        for i in np.arange(self.count):
            low = self.load_images_transform(self.train_low_data_names[i])
            high = self.load_images_transform(self.train_high_data_names[i])
            self.low_data.append(low)
            self.high_data.append(high)

            low_im_filter_max = low[:, :, 2:]
            high_im_filter_max = high[:, :, 2:]
            xxx, bins_of_im = np.histogram(low_im_filter_max, bins=int(self.nbins - 2),
                                           range=(np.min(low_im_filter_max), np.max(low_im_filter_max)))
            hist_c = np.reshape(xxx, [1, 1, nbins - 2])
            hist = np.zeros([1, 1, int(self.nbins + 1)])
            hist[:, :, 0:nbins - 2] = np.array(hist_c, dtype=np.float32) / np.sum(hist_c)
            hist[:, :, nbins - 2:nbins - 1] = np.min(low_im_filter_max)
            hist[:, :, nbins - 1:nbins] = np.max(low_im_filter_max)
            hist[:, :, -1] = high_im_filter_max.mean()

            self.hist_data.append(hist)

    def load_images_transform(self, file):

        data_lowlight = Image.open(file)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        return data_lowlight

    def __getitem__(self, index):

        low = self.low_data[index]
        high = self.high_data[index]
        hist = self.hist_data[index]

        if self.is_train:
            h = low.shape[0]
            w = low.shape[1]

            h_offset = random.randint(0, max(0, h - self.batch_h - 1))
            w_offset = random.randint(0, max(0, w - self.batch_w - 1))

            if self.task != 'test':
                low = low[h_offset:h_offset + self.batch_h, w_offset:w_offset + self.batch_w]
                high = high[h_offset:h_offset + self.batch_h, w_offset:w_offset + self.batch_w]
            else:
                hist[:, :, -1] = self.exp_mean

        img_name = self.train_low_data_names[index].split('\\')[-1]

        return torch.from_numpy(low).float().permute(2, 0, 1), \
            torch.from_numpy(high).float().permute(2, 0, 1), \
            torch.from_numpy(hist).float().permute(2, 0, 1), \
            img_name

    def __len__(self):
        return self.count
