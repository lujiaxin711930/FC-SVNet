import torch
import torch.nn as nn
import math
from torch.autograd import Variable

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
        print(x1.shape)

        out = torch.cat((x1, x2, x3, x4), dim=1)
        print(out.shape)

        return out

image = torch.randn(32, 128, 64, 64)
# out = SPBlock(3, 64)
# print(out)
mgc = MGCBlock(128)
output = mgc(image)
print(output.shape)