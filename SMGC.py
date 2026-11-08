import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class SPBlock(nn.Module):#Strip pooling
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
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        # x=x1+x2
        x = self.conv3(x).sigmoid()
        return x

class MGCBlock(nn.Module):
    def __init__(self, inplans, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(MGCBlock, self).__init__()
        outplanes=inplans
        self.conv_1 = nn.Conv2d(inplans, outplanes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans, outplanes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans, outplanes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans, outplanes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        out = torch.cat((x1, x2, x3, x4), dim=1)

        return out

class SMGC(nn.Module):
    def __init__(self, inplanes):
        super(SMGC, self).__init__()
        self.sp=SPBlock(inplanes)
        self.mgc=MGCBlock(inplanes)
        self.conv1=nn.Conv2d(inplanes,inplanes,1)
        self.conv2 = nn.Conv2d(inplanes*3, inplanes, 1)


    def forward(self, x):
        x1 = self.sp(x)
        x2 = self.mgc(x)
        x3 = self.conv1(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.conv2(out)
        return out


image = Variable(torch.randn(32, 128, 64, 64))
smgc = SMGC(128)
output = smgc(image)
print(output.shape)
