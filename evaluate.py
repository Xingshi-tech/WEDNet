import os

import numpy as np
import torchvision.transforms
from PIL import Image
from cv2 import imread
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import compare_mae

psnr = PeakSignalNoiseRatio().cuda()
ssim = StructuralSimilarityIndexMeasure().cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

enhance_path = './result/'
highRef_path = 'F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/eval15/high/'

files = os.listdir(highRef_path)
files.sort(key=lambda x: int(x.split('.')[0]))

to_tensor = torchvision.transforms.ToTensor()

ssim_val = 0.0
psnr_val = 0.0
lpips_val = 0.0
mae_val = 0.0

total_ssim_val = 0.0
total_psnr_val = 0.0
total_lpips_val = 0.0
total_mae_val = 0.0

for file in tqdm(files):
    highRef_tensor = to_tensor(Image.open(highRef_path + file)).unsqueeze(0).cuda()
    enhance_tensor = to_tensor(Image.open(enhance_path + file)).unsqueeze(0).cuda()

    ssim_val = ssim(enhance_tensor, highRef_tensor)
    psnr_val = psnr(enhance_tensor, highRef_tensor)
    lpips_val = lpips(enhance_tensor, highRef_tensor)

    total_ssim_val += ssim_val
    total_psnr_val += psnr_val
    total_lpips_val += lpips_val

    img_gt = (imread(f'{highRef_path}{file}') / 255.0).astype(np.float32)
    img_pred = (imread(f'{enhance_path}{file}') / 255.0).astype(np.float32)

    mae_val = compare_mae(img_gt, img_pred)
    total_mae_val += mae_val

print(
    f'SSIM:{total_ssim_val / len(files):.4f},PSNR:{total_psnr_val / len(files):.4f},LPIPS:{total_lpips_val / len(files):.4f},MAE:{total_mae_val / len(files):.4f}')
