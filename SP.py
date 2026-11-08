import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class SPBlock(nn.Module):
    def __init__(self, inplanes):
        super(SPBlock, self).__init__()
        outplanes = inplanes
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(outplanes, outplanes, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        print(x1.shape)
        x1 = self.conv1(x1)
        print(x1.shape)
        x1 = self.bn1(x1)
        print(x1.shape)
        x1 = x1.expand(-1, -1, h, w)
        print(x1.shape)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        print(x2.shape)
        x2 = self.conv2(x2)
        print(x2.shape)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        print(x2.shape)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        # x=x1+x2
        x = self.conv3(x).sigmoid()
        print(x.shape)
        return x


image = Variable(torch.randn(32, 3, 64, 64))
# out = SPBlock(3, 64)
# print(out)
SP = SPBlock(3)
output = SP(image)
print(output.shape)

'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class SPM(nn.Module):
    def __init__(self, input, in_channel, out_channel):
        super(SPM, self).__init__()
        inter_channel = in_channel / 4
        b, c, h, w = input.shape

        self.conv1 = nn.Sequential(F.conv2d(in_channel, inter_channel), nn.BatchNorm2d(inter_channel), nn.ReLU(True))

        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.w_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.expand = F.upsample_bilinear((h, w))

        self.conv2 = nn.Sequential(F.conv2d(inter_channel, out_channel), nn.BatchNorm2d(out_channel), nn.ReLU(True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_h = self.conv1(x)
        x_w = self.conv1(x)

        # split
        x_h = self.h_pool(x_h)
        x_w = self.w_pool(x_w)

        # expand
        x_h = self.expand(x_h)
        x_w = self.expand(x_w)

        # fusion
        fusion = x_h + x_w
        fusion = self.conv2(fusion)

        fusion = self.sigmoid(fusion)

        x = x.mul(fusion)

        return F.relu_(x)

if __name__=='main':
    image = Variable(torch.randn(4, 3, 244, 244))
    out = SPM(image, 3, 64)
    print(out)
'''
