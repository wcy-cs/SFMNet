from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
import glob
import random


class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        # 返回指定路径下的文件和文件夹列表。
        self.args = args
        self.train = train
        self.imgs_HR_path = os.path.join(root, 'HR')


        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR_bicubic')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4_bicubic')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16_bicubic')


        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )
        self.imgs_HR = sorted(
                glob.glob(os.path.join(self.imgs_HR_path, '*.png')))

        self.transform = transforms.ToTensor()
        self.train = train
        print(self.imgs_LR_path, self.imgs_HR_path)


    def __getitem__(self, item):

        img_path_LR = os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])

        LR = Image.open(img_path_LR)
        HR = Image.open(img_path_HR)

        HR = np.array(HR)
        LR = np.array(LR)

        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)
        filename = os.path.basename(img_path_HR)

        return {'lr_up': LR, 'img_gt': HR, 'img_name': filename}

    def __len__(self):
        return len(self.imgs_HR)

