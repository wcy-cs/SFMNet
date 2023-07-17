import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.metric import torch_psnr
from utils import util
import torchvision

import models
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

args.epoch = 1000
epochs = args.epochs
model = models.get_model(args)
print(get_parameter_number(model))
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset.Data(root=os.path.join(args.data_path, "CelebA/train"), args=args, train=True)
valdata = dataset.Data(root=os.path.join(args.data_path,'CelebA/val'), args=args, train=False)
testdata1 = dataset.Data(root=os.path.join(args.data_path,'CelebA/test'), args=args, train=False)
testdata2 = dataset.Data(root=os.path.join(args.data_path,'helen/test'), args=args, train=False)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=32)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)
testset1 = DataLoader(testdata1, batch_size=1, shuffle=False, num_workers=1)
testset2 = DataLoader(testdata2, batch_size=1, shuffle=False, num_workers=1)


class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)


class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)

optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
device = torch.device(args.device)
def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample

def eval_model(model, dataset, name, epoch, args):
    model.eval()
    val_psnr_dic = 0
    val_ssim_dic = 0
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    timer_test = util.timer()
    for batch, data in enumerate(dataset):

        sr = model(to_device(data, device))
        psnr_c, ssim_c = torch_psnr(data['img_gt'], sr['img_out'])
        val_psnr_dic = val_psnr_dic + psnr_c
        val_ssim_dic = val_ssim_dic + ssim_c
    print("Epoch：{}, {}, psnr: {:.3f}".format(epoch+1, name, val_psnr_dic/(len(dataset))))
    print('Forward: {:.2f}s\n'.format(timer_test.toc()))
    writer.add_scalar("{}_psnr_DIC".format(name), val_psnr_dic/len(dataset), epoch)
    writer.add_scalar("{}_ssim_DIC".format(name), val_ssim_dic / len(dataset), epoch)

def train_model(model, trainset, epoch, args):
    # print(model)
    model.train()
    train_loss = 0
    criterion1 = nn.L1Loss().to(device, non_blocking=True)
    amploss = AMPLoss().to(device, non_blocking=True)
    phaloss = PhaLoss().to(device, non_blocking=True)
    for batch, data in enumerate(trainset):
        sr = model(to_device(data, device))
        loss = criterion1(sr['img_out'], data['img_gt']) + \
                       args.fft_weight * amploss(sr['img_fre'], data['img_gt']) + args.fft_weight * phaloss(
                    sr['img_fre'],
                    data[
                        'img_gt']) + \
                       criterion1(sr['img_fre'], data['img_gt'])
        optimizer.zero_grad()
        loss.backward()
        train_loss = train_loss + loss.item()
        optimizer.step()
    print("Epoch：{} loss: {:.3f}".format(epoch+1, train_loss/(len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss /(len(trainset)) * 255, i)

    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'model'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, args.writer_name, 'model', 'epoch{}.pth'.format(epoch + 1)))

for i in range(epochs):
    train_model(model, trainset, i, args)
    eval_model(model, valset, "val", i, args)
    eval_model(model, testset1, "CelebA", i, args)
    eval_model(model, testset2, "Helen", i, args)












