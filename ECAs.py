import torch
import torch.nn as nn
import math
from torchsummary import  summary

# ----------------------------#
# ECAs channel attention
# ----------------------------#
class ECSs(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECSs, self).__init__()
        # ----------------------------------#
        # Determine the adaptive convolution kernel size from the channel count
        # ----------------------------------#
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------#
        # Combine global average and max pooling before channel-wise convolution,
        # then use sigmoid activation to obtain a weight for each channel,
        # and rescale the input feature map with these channel weights.
        # ------------------------------------------#
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = torch.add(y1, y2)
        y = torch.sigmoid(y)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        print(y.shape)
        y = self.sigmoid(y)
        print(y.shape)
        return x * y.expand_as(x)

if __name__ == '__main__':
    input = torch.randn(32, 64, 224, 224)
    ecas = ECSs(channel=64)
    output = ecas(input)
    print(output.shape)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = eca_block(channel=64).to(device)
    # # vgg=models.vgg16().to(device)
    # summary(model, (64, 224, 224))
    # print(summary(model, (64, 224, 224)))