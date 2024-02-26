import os

import numpy as np
import torch
import torch.optim
from PIL import Image, ImageEnhance
from tqdm import tqdm

from MyModel import enhance_net_nopool
from utils import tensor2img


def post_processing(im, dest_folder):
    contrasted = ImageEnhance.Contrast(im).enhance(1.03)
    colored = ImageEnhance.Color(contrasted).enhance(1.37)
    brighted = ImageEnhance.Brightness(colored).enhance(0.97)
    sharped = ImageEnhance.Sharpness(brighted).enhance(0.90)
    sharped.save(dest_folder)


def lowlight(image_path, image_high_name, save_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 20
    nbins = 14

    data_lowlight = Image.open(image_path)

    data_highlight = (np.asarray(Image.open(image_high_name)) / 255.0)

    exp_mean = np.max(data_highlight, axis=2, keepdims=True).mean()

    data_highlight = torch.from_numpy(data_highlight).float().permute(2, 0, 1).cuda().unsqueeze(0)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    low_im_filter_max = np.max(data_lowlight, axis=2, keepdims=True)  # positive
    hist = np.zeros([1, 1, int(nbins + 1)])

    xxx, bins_of_im = np.histogram(low_im_filter_max, bins=int(nbins - 2),
                                   range=(np.min(low_im_filter_max), np.max(low_im_filter_max)))

    hist_c = np.reshape(xxx, [1, 1, nbins - 2])
    hist[:, :, 0:nbins - 2] = np.array(hist_c, dtype=np.float32) / np.sum(hist_c)
    hist[:, :, nbins - 2:nbins - 1] = np.min(low_im_filter_max)
    hist[:, :, nbins - 1:nbins] = np.max(low_im_filter_max)
    hist[:, :, -1] = exp_mean

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    hist = torch.from_numpy(hist).float().permute(2, 0, 1).cuda().unsqueeze(0)

    wednet = enhance_net_nopool(nbins).cuda()
    wednet.load_state_dict(torch.load('./snapshots/best_epoch.pth'))
    enhanced_image = wednet(data_lowlight, hist)

    result_path = save_path + image_path.split('/')[-1]
    enhanced_image = enhanced_image[0]

    im = tensor2img(enhanced_image)
    post_processing(im, result_path)


if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/eval15/low/'
        filePath_high = 'F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/eval15/high/'

        save_path = './result/'

        file_list = os.listdir(filePath)
        file_list.sort(key=lambda x: int(x.split('.')[0]))
        for file_name in tqdm(file_list):
            image_name = filePath + file_name
            image_high_name = filePath_high + file_name
            lowlight(image_name, image_high_name, save_path)
