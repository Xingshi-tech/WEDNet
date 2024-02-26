import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def tensor2img(tensor_image):
    tensor = tensor_image.cpu().detach().numpy()
    tensor = (tensor * 255).clip(0, 255)
    tensor = tensor.astype(np.uint8)
    tensor = tensor.transpose((1, 2, 0))
    pil_image = TF.to_pil_image(tensor)

    return pil_image


def img2tensor(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img) / 255.0
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

    return tensor, img

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
