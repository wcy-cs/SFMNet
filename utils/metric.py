
import math
import numpy as np
from skimage.metrics import structural_similarity


def quantize(img):
    return (img *  255).clip(0, 255).round()

def tensor2img(tensor):
    return quantize(tensor.detach().cpu().numpy()).astype(np.uint8).transpose(0, 2, 3, 1)

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    #
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    mse = np.mean((img1_np - img2_np)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def torch_psnr(img_pred, img_gt):
    img_gt = tensor2img(img_gt)
    img_pred = tensor2img(img_pred)

    img_pred, img_gt = img_pred[:, 8: -8, 8:-8, :], img_gt[:, 8: -8, 8:-8, :]

    sum_psnr = []
    sum_ssim = []
    for i in range(img_gt.shape[0]):
        sum_psnr.append(calc_psnr(rgb2ycbcr(img_pred[i]), rgb2ycbcr(img_gt[i])))
        sum_ssim.append(structural_similarity(rgb2ycbcr(img_pred[i]), rgb2ycbcr(img_gt[i]), data_range=255))
    return np.mean(sum_psnr),  np.mean(sum_ssim)