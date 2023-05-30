import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name

import torch
import torch.optim as optim
import os
import torch.nn as nn
from loss import gan
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from option import args
import torchvision
import models
from utils.metric import torch_psnr
from models.discriminator import LightCNN_9Layers, VGGFeatureExtractor, LightCNN_9Layers_fft

device = torch.device(args.device)
def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample


epochs = args.epochs
start_epoch = 0
lr = args.lr
model = models.get_model(args)
D = LightCNN_9Layers()
Vgg = VGGFeatureExtractor()
fftD = LightCNN_9Layers_fft()
fftD = fftD.to(device, non_blocking=True)
D = D.to(device, non_blocking=True)
Vgg = Vgg.to(device, non_blocking=True)
Vgg.eval()
pre_trained = torch.load(args.load)
model.load_state_dict(pre_trained)
model = model.to(device, non_blocking=True)

writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset.Data(root=os.path.join(args.data_path,"CelebA/train"), args=args, train=True)
valdata = dataset.Data(root=os.path.join(args.data_path,'CelebA/val'), args=args, train=False)
testdata1 = dataset.Data(root=os.path.join(args.data_path,'CelebA/test'), args=args, train=False)
testdata2 = dataset.Data(root=os.path.join(args.data_path,'helen/test'), args=args, train=False)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=32)
valset = DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=1)
testset1 = DataLoader(testdata1, batch_size=args.batch_size, shuffle=False, num_workers=1)
testset2 = DataLoader(testdata2, batch_size=args.batch_size, shuffle=False, num_workers=1)
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


def eval_model(model, dataset, name, epoch, args):
    model.eval()
    val_psnr_dic = 0
    val_ssim_dic = 0
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    for batch, data in enumerate(dataset):

        sr = model(to_device(data, device))
        psnr_c, ssim_c = torch_psnr(data['img_gt'], sr['img_out'])
        val_psnr_dic = val_psnr_dic + psnr_c
        val_ssim_dic = val_ssim_dic + ssim_c

    print("Epoch：{}, {}, psnr: {:.3f}".format(epoch+1, name, val_psnr_dic/(len(dataset))))
    writer.add_scalar("{}_psnr_DIC".format(name), val_psnr_dic/len(dataset), epoch)
    writer.add_scalar("{}_ssim_DIC".format(name), val_ssim_dic / len(dataset), epoch)

def train_model(model, trainset, epoch, args):
    model.train()
    criterion1 = nn.L1Loss().to(device, non_blocking=True)
    amploss = AMPLoss().to(device, non_blocking=True)
    phaloss = PhaLoss().to(device, non_blocking=True)
    Gan_loss = gan.GANLoss().to(device, non_blocking=True)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)
    optimizer_D = optim.Adam(params=D.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)
    optimizer_D_fft = optim.Adam(params=fftD.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)

    train_loss = 0
    d_loss = 0
    d_loss_fft = 0
    for batch, data in enumerate(trainset):
        sr = model(to_device(data, device))

        l1_loss = criterion1(sr['img_out'], data['img_gt']) + \
                      args.fft_weight * amploss(sr['img_fre'], data['img_gt']) + args.fft_weight * phaloss(
                sr['img_fre'],
                data[
                    'img_gt']) + \
                      criterion1(sr['img_fre'], data['img_gt'])

        SR_feature = Vgg(sr['img_out']).detach()
        HR_feature = Vgg(data['img_gt']).detach()
        perceptual_loss = criterion1(HR_feature, SR_feature)

        SR_pred = D(sr['img_out'])
        # HR_pred = D(data['img_gt']).detach()
        loss_g_GAN = Gan_loss(SR_pred, True)

        SR_pred_fft = fftD(sr['img_out'])
        loss_g_GAN_fft = Gan_loss(SR_pred_fft, True)
        loss_g = 0.0005 * loss_g_GAN + l1_loss  + 0.1 * perceptual_loss + args.fftd_weight * loss_g_GAN_fft

        train_loss = train_loss + loss_g.item()
        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()
        optimizer_D.zero_grad()
        optimizer_D_fft.zero_grad()


        for p in D.parameters():
            p.requires_grad = True
        for p in fftD.parameters():
            p.requires_grad = True

        HR_pred = D(data['img_gt'])
        SR_pred = D(sr['img_out'].detach())
        loss_d = Gan_loss(HR_pred, True) + Gan_loss(SR_pred, False)
        d_loss += loss_d.item()
        loss_d.backward()
        optimizer_D.step()

        SR_pred_fft = fftD(sr['img_out']).detach()
        HR_pred_fft = fftD(data['img_gt'])
        loss_d_fft = args.fftd_weight * Gan_loss(HR_pred_fft, True) + args.fftd_weight * Gan_loss(SR_pred_fft, False)

        d_loss_fft += loss_d_fft.item()
        loss_d_fft.backward()
        optimizer_D_fft.step()



    print("Epoch：{} loss: {:.3f}".format(epoch + 1, train_loss / (len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss / (len(trainset)) * 255, epoch + 1)
    writer.add_scalar('d_loss', d_loss / (len(trainset)) * 255, epoch + 1)
    writer.add_scalar('d_loss_fft', d_loss_fft / (len(trainset)) * 255, epoch + 1)
    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'model'), exist_ok=True)
    torch.save(model.state_dict(),
                   os.path.join(args.save_path, args.writer_name, 'model', 'epoch{}.pth'.format(epoch + 1)))

if __name__ == "__main__":
    for i in range(epochs):
        train_model(model, trainset, i, args)
        eval_model(model, valset, "val", i, args)
        eval_model(model, testset1, "CelebA", i, args)
        eval_model(model, testset2, "Helen", i, args)















