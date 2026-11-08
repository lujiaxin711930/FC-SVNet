import torch
import torch.nn as nn
import math

# ----------------------------#
# Efficient Channel Attention (ECA)
# ----------------------------#
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        # ----------------------------------#
        # Compute the adaptive convolution kernel size from the channel count
        # ----------------------------------#
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------#
        # Apply global average pooling followed by channel-wise convolution,
        # then use sigmoid activation to obtain a weight for each channel,
        # and rescale the input feature map with these channel weights.
        # ------------------------------------------#
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

if __name__ == '__main__':
    input = torch.randn(32, 512, 64, 64)
    eca = eca_block(channel=512)
    output = eca(input)
    print(output.shape)