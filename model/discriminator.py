import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F
# Assume input range is [0, 1], RGB
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=False,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # x: [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(3, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.classifier = nn.Sequential(mfm(8*8*128, 256, type=0), nn.LeakyReLU(0.2, True),
                nn.Linear(256, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.classifier(x)
        return out

class network_9layers_fft(nn.Module):
    def __init__(self):
        super(network_9layers_fft, self).__init__()
        self.features = nn.Sequential(
            mfm(6, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.classifier = nn.Sequential(mfm(5120, 256, type=0), nn.LeakyReLU(0.2, True),
                nn.Linear(256, 1))

    def forward(self, x):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.abs(x)
        x_pha = torch.angle(x)
        # print(x_mag.shape, y_mag.shape)
        x = self.features(torch.cat((x_mag, x_pha), 1))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.classifier(x)
        return out

class feature_extractor_9layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(
            nn.Sequential(mfm(3, 48, 5, 1, 2), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(48, 96, 3, 1, 1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(96, 192, 3, 1, 1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )
        self.features.append(
            nn.Sequential(group(192, 128, 3, 1, 1),
                group(128, 128, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        )

    def forward(self, x):
        feature_list = []
        for f in self.features:
            x = f(x)
            feature_list.append(x)
        return feature_list

def LightCNN_Feature_9Layers(**kwargs):
    model = feature_extractor_9layers(**kwargs)
    return model

def LightCNN_9Layers(**kwargs):
    model = network_9layers()
    return model

def LightCNN_9Layers_fft(**kwargs):
    model = network_9layers_fft()
    return model
