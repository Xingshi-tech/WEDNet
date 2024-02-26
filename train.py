import argparse
import os
import time
from datetime import datetime

import pytz
import torch
import torch.optim
from torch.utils.data import DataLoader

import Myloss
import wandb
from MyModel import enhance_net_nopool
from Myloss import validation
from dataloader import MemoryFriendlyLoader_zy

PSNR_mean = 0
SSIM_mean = 0

beijing_tz = pytz.timezone('Asia/Shanghai')

run_name = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
wandb.init(
    project='WEDNet',
    name=f'train-{run_name}'
)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    wednet = enhance_net_nopool(config.nbins).cuda()

    if config.load_pretrain == True:
        wednet.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = MemoryFriendlyLoader_zy(low_img_dir=config.lowlight_images_path,
                                            high_img_dir=config.highlight_images_path, task=config.task,
                                            batch_w=config.patch_size, batch_h=config.patch_size,
                                            nbins=config.nbins, exp_mean=config.exp_mean, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    val_dataset = MemoryFriendlyLoader_zy(low_img_dir=config.val_lowlight_images_path,
                                          high_img_dir=config.val_highlight_images_path, task=config.task,
                                          batch_w=50, batch_h=config.patch_size,
                                          nbins=config.nbins, exp_mean=config.exp_mean, is_train=False)

    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                            num_workers=config.num_workers,
                            pin_memory=True)

    L_color_zy = Myloss.L_color_zy()
    L_grad_cosist = Myloss.L_grad_cosist()
    L_bright_cosist = Myloss.L_bright_cosist()
    L_recon = Myloss.L_recon()
    L_brightness = Myloss.L_brightness()

    optimizer = torch.optim.Adam(wednet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wednet.train()
    ssim_high = 0
    psnr_high = 0

    start_time = time.time()
    current_ten_take_time = start_time
    since_update = -1
    for epoch in range(config.num_epochs):
        since_update += 1
        for iteration, (img_lowlight, img_highlight, hist, img_name) in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            img_highlight = img_highlight.cuda()
            hist = hist.cuda()

            enhanced_image = wednet(img_lowlight, hist)

            Loss_1 = L_grad_cosist(enhanced_image, img_highlight)
            Loss_6 = L_bright_cosist(enhanced_image, img_highlight)
            loss_2, Loss_ssim = L_recon(enhanced_image, img_highlight)
            loss_col = torch.mean(L_color_zy(enhanced_image, img_highlight))
            loss_bri = L_brightness(enhanced_image, img_highlight)

            loss = Loss_ssim + loss_2 + Loss_1 + Loss_6 + loss_col + loss_bri
            wandb.log({
                "Loss_1": Loss_1.item(), "Loss_6": Loss_6.item(), "loss_2": loss_2.item(),
                "Loss_ssim": Loss_ssim.item(), "loss_col": loss_col.item(), "loss_bri": loss_bri.item(),
                "total_loss": loss.item()
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.display_iter == 0:
            print(f"Since last update:{since_update}")
            wednet.eval()
            SSIM_mean, PSNR_mean = validation(wednet, val_loader)

            wandb.log({
                "epoch": epoch, "SSIM": SSIM_mean, "PSNR": PSNR_mean
            })

            if SSIM_mean > ssim_high or PSNR_mean > psnr_high:
                wandb.log({
                    "since_update": since_update
                })

            if SSIM_mean > ssim_high:
                ssim_high = SSIM_mean
                print('the highest SSIM value is:{:5,.3f}'.format(ssim_high))
                torch.save(wednet.state_dict(), os.path.join(config.snapshots, "best_epoch" + '.pth'))
                since_update = 0
                wandb.log({'max_ssim': ssim_high})
            if PSNR_mean > psnr_high:
                psnr_high = PSNR_mean
                print('the highest PSNR value is:{:6,.3f}'.format(psnr_high))
                torch.save(wednet.state_dict(), os.path.join(config.snapshots, "best_epoch" + '.pth'))
                since_update = 0
                wandb.log({'max_psnr': psnr_high})
            print(
                'epoch/total:{:5}/{},total take time:{:9,.3f}s,take time:{:6,.3f}s'.format(epoch, config.num_epochs,
                                                                                           time.time() - start_time,
                                                                                           time.time() - current_ten_take_time))
            print('-----------------------------------------------------------')
            current_ten_take_time = time.time()

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default="F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/our485/low/")
    parser.add_argument('--highlight_images_path', type=str,
                        default="F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/our485/high/")
    parser.add_argument('--val_lowlight_images_path', type=str,
                        default="F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/eval15/low/")
    parser.add_argument('--val_highlight_images_path', type=str,
                        default="F:/Acodes/myPytorchCodes/0_datasets/LOL-v1/eval15/high/")

    parser.add_argument('--task', type=str, default="train")
    parser.add_argument('--nbins', type=int, default=14)
    parser.add_argument('--patch_size', type=int, default=100)
    parser.add_argument('--exp_mean', type=float, default=0.55)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=15000)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshots', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots):
        os.mkdir(config.snapshots)

    train(config)
